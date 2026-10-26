//! Mask-local, chroma-preserving 2D portrait lighting (not 3D relighting).
//!
//! Buffers contain normalized, unpremultiplied Gray/GrayA/RGB/RGBA samples.
//! No colour-space conversion, spatial filtering, detection or geometry edits
//! are performed. Subject lift and virtual light use `person_mask`; the nose
//! contour uses only `skin_mask` and valid eye/nose/mouth landmarks. Callers
//! supply those masks (including any desired feathering).
//!
//! RGB always receives one shared gain: hue and RGB ratios/HSV saturation are
//! preserved, without independently clipping channels. Existing black stays
//! black; invalid or out-of-range colour pixels are left untouched. Alpha,
//! inactive pixels (including signed zero/NaN payloads), and zero-strength
//! settings are copied bit-for-bit. This is a display-range enhancement, not
//! an HDR tone mapper or a reconstruction of illumination/normals/occlusion.

use crate::Error;
use crate::geom::{FaceGeom, Point};

/// Independent strengths and image-relative virtual-light controls, all 0..1.
///
/// Finite values are clamped. Non-finite strengths become zero; non-finite
/// positions and size become 0.5. Size zero is a small soft source, not off;
/// `virtual_light = 0` switches that effect off. X grows rightward, Y downward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelightSpec {
    pub subject_brighten: f32,
    pub virtual_light: f32,
    pub light_x: f32,
    pub light_y: f32,
    pub light_size: f32,
    pub nose_contour: f32,
}

impl Default for RelightSpec {
    fn default() -> Self {
        Self { subject_brighten: 0.0, virtual_light: 0.0, light_x: 0.5, light_y: 0.5, light_size: 0.5, nose_contour: 0.0 }
    }
}

fn unit(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, 1.0) } else { fallback }
}

fn copy_image(img: &[f32]) -> Result<Vec<f32>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(img.len()).map_err(|_| Error::InvalidParam("relight output allocation failed".into()))?;
    out.extend_from_slice(img);
    Ok(out)
}

/// Apply independent subject lift, a movable soft 2D light, and nose contour.
///
/// Both masks must have exactly `w * h` elements, even for identity settings.
/// Non-finite mask samples act as zero; other mask samples are clamped to 0..1.
/// Empty/overflowing dimensions, unsupported channel counts and length errors
/// return `Err`. Degenerate, non-finite or implausible nose geometry is skipped.
/// No global mask fallback is used when detections or masks are absent.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    person_mask: &[f32],
    skin_mask: &[f32],
    faces: &[FaceGeom],
    spec: &RelightSpec,
) -> Result<Vec<f32>, Error> {
    if w == 0 || h == 0 || ch == 0 {
        return Err(Error::EmptyInput("relight image"));
    }
    if ch > 4 {
        return Err(Error::InvalidParam("relight requires Gray, GrayA, RGB or RGBA".into()));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam("relight dimensions overflow".into()))?;
    let samples = pixels.checked_mul(ch).ok_or_else(|| Error::InvalidParam("relight sample count overflow".into()))?;
    if samples > (isize::MAX as usize) / core::mem::size_of::<f32>() {
        return Err(Error::InvalidParam("relight buffer exceeds addressable size".into()));
    }
    for (what, expected, found) in
        [("relight image", samples, img.len()), ("relight person mask", pixels, person_mask.len()), ("relight skin mask", pixels, skin_mask.len())]
    {
        if expected != found {
            return Err(Error::ShapeMismatch { what, expected, found });
        }
    }
    let subject = unit(spec.subject_brighten, 0.0);
    let light = unit(spec.virtual_light, 0.0);
    let nose = unit(spec.nose_contour, 0.0);
    let mut out = copy_image(img)?;
    if subject == 0.0 && light == 0.0 && nose == 0.0 {
        return Ok(out);
    }

    let width = w as f32;
    let height = h as f32;
    let min_dim = width.min(height);
    let light_x = unit(spec.light_x, 0.5) * width;
    let light_y = unit(spec.light_y, 0.5) * height;
    // A circular footprint in pixel space, including on non-square images.
    let radius = (0.15 + 1.35 * unit(spec.light_size, 0.5)) * min_dim;
    let mut noses = Vec::new();
    if nose > 0.0 {
        noses.try_reserve_exact(faces.len()).map_err(|_| Error::InvalidParam("relight face allocation failed".into()))?;
        noses.extend(faces.iter().filter_map(|face| NoseFrame::new(face, width, height)));
    }
    let color_ch = if ch >= 3 { 3 } else { 1 };
    for (i, (((src, dst), person), skin)) in img.chunks_exact(ch).zip(out.chunks_exact_mut(ch)).zip(person_mask).zip(skin_mask).enumerate() {
        let person = unit(*person, 0.0);
        let skin = unit(*skin, 0.0);
        if (person == 0.0 || (subject == 0.0 && light == 0.0)) && (skin == 0.0 || noses.is_empty()) {
            continue;
        }
        let Some(colors) = src.get(..color_ch) else { continue };
        // Do not turn invalid/HDR source pixels into normalized/clipped data.
        if colors.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            continue;
        }
        let peak = colors.iter().copied().fold(0.0_f32, f32::max);
        if peak == 0.0 {
            continue;
        }
        let luma = if color_ch == 3 { colors.iter().zip([0.2126, 0.7152, 0.0722]).map(|(v, weight)| v * weight).sum() } else { peak };
        let x = (i % w) as f32 + 0.5;
        let y = (i / w) as f32 + 0.5;
        let field = if person > 0.0 && light > 0.0 { bump(((x - light_x) / radius).hypot((y - light_y) / radius)) } else { 0.0 };
        // Take the strongest signed contour, rather than accumulating duplicate
        // detections. This also bounds the combined gain for highlight safety.
        let contour = if skin > 0.0 && nose > 0.0 {
            noses.iter().map(|frame| frame.contour(x, y)).fold(0.0_f32, |a, b| if b.abs() > a.abs() { b } else { a }) * skin * nose
        } else {
            0.0
        };
        // Multiplicative lift favours shadows/midtones. Peak-channel headroom
        // protects both neutral highlights and saturated colour highlights.
        // Positive <= 1.8 + 1.5 + 0.24 < 4, so for peak p in [0,1],
        // p * (1 + positive * (1-p)^2) <= 1 (since p*(1-p) <= 1/4).
        // No output-channel clamp is needed, preserving chromatic ratios.
        let positive = person * (1.8 * subject * (1.0 - luma) + 1.5 * light * field) + 0.24 * contour.max(0.0);
        let negative = 0.10 * contour.min(0.0);
        let gain = 1.0 + positive * (1.0 - peak).powi(2) + negative;
        if gain == 1.0 {
            continue;
        }
        for (value, original) in dst.iter_mut().zip(colors).take(color_ch) {
            *value = *original * gain;
        }
    }
    Ok(out)
}

/// Compact C1-continuous falloff, exactly zero outside its support.
fn bump(t: f32) -> f32 {
    let q = (1.0 - t * t).max(0.0);
    q * q
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Clone, Copy)]
struct NoseFrame {
    origin: Point,
    across: Point,
    down: Point,
    eye_distance: f32,
    depth: f32,
    tip_across: f32,
    eyes: [Point; 2],
    eye_radius: f32,
    mouth: [Point; 3],
}

impl NoseFrame {
    fn new(face: &FaceGeom, w: f32, h: f32) -> Option<Self> {
        let points = [face.left_eye, face.right_eye, face.nose_tip, face.mouth_left, face.mouth_right, face.mouth_center];
        if points.iter().any(|p| !p.is_valid() || !(0.0..=1.0).contains(&p.x) || !(0.0..=1.0).contains(&p.y)) {
            return None;
        }
        let pixel = |p: Point| Point { x: p.x * w, y: p.y * h };
        let left = pixel(face.left_eye);
        let right = pixel(face.right_eye);
        let tip = pixel(face.nose_tip);
        let dx = right.x - left.x;
        let dy = right.y - left.y;
        let eye_distance = dx.hypot(dy);
        if eye_distance < 1e-3 {
            return None;
        }
        let across = Point { x: dx / eye_distance, y: dy / eye_distance };
        let mut down = Point { x: -across.y, y: across.x };
        let origin = Point { x: (left.x + right.x) * 0.5, y: (left.y + right.y) * 0.5 };
        let tip_x = (tip.x - origin.x) / eye_distance;
        let tip_y = (tip.y - origin.y) / eye_distance;
        let depth = (tip_x * down.x + tip_y * down.y).abs();
        let tip_across = tip_x * across.x + tip_y * across.y;
        if !(0.15..=1.5).contains(&depth) || tip_across.abs() > 0.65 {
            return None;
        }
        if tip_x * down.x + tip_y * down.y < 0.0 {
            down.x = -down.x;
            down.y = -down.y;
        }
        let eye_radius =
            if face.eye_radius.is_finite() { (face.eye_radius * w.min(h)).clamp(eye_distance * 0.20, eye_distance * 0.40) } else { eye_distance * 0.25 };
        Some(Self {
            origin,
            across,
            down,
            eye_distance,
            depth,
            tip_across,
            eyes: [left, right],
            eye_radius,
            mouth: [pixel(face.mouth_left), pixel(face.mouth_right), pixel(face.mouth_center)],
        })
    }

    fn contour(&self, x: f32, y: f32) -> f32 {
        let dx = (x - self.origin.x) / self.eye_distance;
        let dy = (y - self.origin.y) / self.eye_distance;
        let t = (dx * self.down.x + dy * self.down.y) / self.depth;
        if !(0.12..1.10).contains(&t) {
            return 0.0;
        }
        // Shear the rotated eye-axis frame toward the actual nose tip, so a
        // slightly off-centre nose does not get a straight misplaced stripe.
        let u = dx * self.across.x + dy * self.across.y - self.tip_across * t;
        if u.abs() >= 0.31 {
            return 0.0;
        }
        let vertical = smoothstep((t - 0.12) / 0.22) * (1.0 - smoothstep((t - 0.88) / 0.22));
        let mut protect = 1.0;
        for eye in self.eyes {
            protect *= smoothstep(((x - eye.x).hypot(y - eye.y) / self.eye_radius - 1.0) / 0.35);
        }
        let [left, right, center] = self.mouth;
        let lip_distance = segment_distance(x, y, left, right).min((x - center.x).hypot(y - center.y));
        protect *= smoothstep((lip_distance / (self.eye_distance * 0.14) - 1.0) / 0.4);
        let bridge = bump(u / 0.12);
        let sides = bump((u.abs() - 0.21) / 0.10);
        vertical * protect * (bridge - sides)
    }
}

fn segment_distance(x: f32, y: f32, start: Point, end: Point) -> f32 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let len = dx.hypot(dy);
    if len <= 1e-6 {
        return (x - start.x).hypot(y - start.y);
    }
    let ux = dx / len;
    let uy = dy / len;
    let along = ((x - start.x) * ux + (y - start.y) * uy).clamp(0.0, len);
    (x - start.x - along * ux).hypot(y - start.y - along * uy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn face() -> FaceGeom {
        FaceGeom::from_sparse(
            [0.1, 0.1, 0.9, 0.95],
            Point { x: 0.3, y: 0.3 },
            Point { x: 0.7, y: 0.3 },
            Point { x: 0.5, y: 0.58 },
            Point { x: 0.38, y: 0.73 },
            Point { x: 0.62, y: 0.73 },
        )
        .unwrap()
    }

    #[test]
    fn zero_strength_is_bit_exact_identity() {
        let img = [0.2, -0.0, f32::from_bits(0x7fc0_0123), 0.6, 1.2, -0.2, 0.7, 0.0];
        let spec = RelightSpec { light_x: 0.0, light_y: 1.0, light_size: 0.0, ..RelightSpec::default() };
        let out = apply(&img, 2, 1, 4, &[1.0; 2], &[1.0; 2], &[face()], &spec).unwrap();
        assert_eq!(bits(&out), bits(&img));
    }

    #[test]
    fn zero_masks_preserve_all_bits_and_do_not_leak() {
        let mut img = vec![0.25; 64 * 64 * 4];
        img[0] = -0.0;
        img[1] = f32::from_bits(0x7fc0_0456);
        let spec = RelightSpec { subject_brighten: 1.0, virtual_light: 1.0, nose_contour: 1.0, ..RelightSpec::default() };
        let mut mask = vec![0.0; 64 * 64];
        let active = 32 * 64 + 32;
        mask[active] = 1.0;
        let out = apply(&img, 64, 64, 4, &mask, &mask, &[face()], &spec).unwrap();
        for i in 0..64 * 64 {
            if i != active {
                assert_eq!(bits(&out[i * 4..i * 4 + 4]), bits(&img[i * 4..i * 4 + 4]));
            }
        }
        assert!(out[active * 4] > img[active * 4]);
        let zeros = vec![0.0; 64 * 64];
        assert_eq!(bits(&apply(&img, 64, 64, 4, &zeros, &zeros, &[face()], &spec).unwrap()), bits(&img));
    }

    #[test]
    fn rgba_and_gray_alpha_are_unchanged() {
        let spec = RelightSpec { subject_brighten: 1.0, virtual_light: 1.0, ..RelightSpec::default() };
        for ch in [2, 4] {
            let mut img = vec![0.2; 4 * ch];
            let alpha = [-0.0, 0.25, f32::from_bits(0x7fc0_0234), 1.0];
            for (pixel, a) in img.chunks_exact_mut(ch).zip(alpha) {
                pixel[ch - 1] = a;
            }
            let out = apply(&img, 2, 2, ch, &[1.0; 4], &[0.0; 4], &[], &spec).unwrap();
            assert!(out[0] > img[0]);
            for (pixel, a) in out.chunks_exact(ch).zip(alpha) {
                assert_eq!(pixel[ch - 1].to_bits(), a.to_bits());
            }
        }
    }

    #[test]
    fn shadows_and_midtones_lift_more_than_highlights() {
        let img = [0.12, 0.4, 0.95, 1.0, 0.0];
        let spec = RelightSpec { subject_brighten: 1.0, ..RelightSpec::default() };
        let out = apply(&img, 5, 1, 1, &[1.0; 5], &[0.0; 5], &[], &spec).unwrap();
        assert!(out[0] - img[0] > 0.08);
        assert!(out[1] - img[1] > 0.08);
        assert!(out[2] - img[2] < 0.001);
        assert_eq!(out[3].to_bits(), img[3].to_bits());
        assert_eq!(out[4].to_bits(), img[4].to_bits());
    }

    #[test]
    fn chromatic_ratios_and_saturation_are_preserved_without_clipping() {
        let img = [0.6, 0.3, 0.15, 0.98, 0.49, 0.245, 1.0, 0.2, 0.0];
        let spec = RelightSpec { subject_brighten: 1.0, virtual_light: 1.0, light_size: 1.0, ..RelightSpec::default() };
        let out = apply(&img, 3, 1, 3, &[1.0; 3], &[0.0; 3], &[], &spec).unwrap();
        assert!(out[0] > img[0]);
        for (src, dst) in img.chunks_exact(3).zip(out.chunks_exact(3)).take(2) {
            assert!(dst[0] < 1.0);
            assert!((dst[1] / dst[0] - src[1] / src[0]).abs() < 1e-6);
            assert!((dst[2] / dst[0] - src[2] / src[0]).abs() < 1e-6);
            assert!(((dst[0] - dst[2]) / dst[0] - (src[0] - src[2]) / src[0]).abs() < 1e-6);
        }
        assert_eq!(&out[6..], &img[6..]);
    }

    #[test]
    fn combined_lifts_never_clip_highlights() {
        let spec = RelightSpec { subject_brighten: 1.0, virtual_light: 1.0, nose_contour: 1.0, light_size: 1.0, ..RelightSpec::default() };
        let mask = vec![1.0; 100 * 100];
        for value in [0.1, 0.4, 0.65, 0.9, 0.99, 0.999_999] {
            let img = vec![value; 100 * 100];
            let out = apply(&img, 100, 100, 1, &mask, &mask, &[face()], &spec).unwrap();
            assert!(out.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        }
    }

    #[test]
    fn virtual_light_position_controls_horizontal_and_vertical_direction() {
        let img = vec![0.3; 21 * 21];
        let mask = vec![1.0; 21 * 21];
        let spec = RelightSpec { virtual_light: 1.0, light_x: 0.2, light_y: 0.5, light_size: 0.6, ..RelightSpec::default() };
        let left = apply(&img, 21, 21, 1, &mask, &mask, &[], &spec).unwrap();
        let right = apply(&img, 21, 21, 1, &mask, &mask, &[], &RelightSpec { light_x: 0.8, ..spec }).unwrap();
        assert!(left[10 * 21 + 4] > left[10 * 21 + 16]);
        assert!(right[10 * 21 + 16] > right[10 * 21 + 4]);
        assert!((left[10 * 21 + 4] - right[10 * 21 + 16]).abs() < 1e-6);
        let top = apply(&img, 21, 21, 1, &mask, &mask, &[], &RelightSpec { light_x: 0.5, light_y: 0.2, ..spec }).unwrap();
        assert!(top[4 * 21 + 10] > top[16 * 21 + 10]);
    }

    #[test]
    fn virtual_light_size_changes_soft_footprint_not_peak_strength() {
        let img = vec![0.3; 41 * 21];
        let mask = vec![1.0; 41 * 21];
        let spec = RelightSpec { virtual_light: 1.0, light_size: 0.0, ..RelightSpec::default() };
        let small = apply(&img, 41, 21, 1, &mask, &mask, &[], &spec).unwrap();
        let large = apply(&img, 41, 21, 1, &mask, &mask, &[], &RelightSpec { light_size: 1.0, ..spec }).unwrap();
        let center = 10 * 41 + 20;
        assert_eq!(small[center], large[center]);
        assert_eq!(small[center + 10].to_bits(), img[center + 10].to_bits());
        assert!(large[center + 10] > img[center + 10]);
        // Equal pixel distances on this non-square image have equal gains.
        assert!((large[center + 8] - large[center + 8 * 41]).abs() < 1e-6);
        assert!(small[center] > small[center + 1] && small[center + 1] > small[center + 2]);
    }

    #[test]
    fn mask_feather_weights_gain_and_parameters_are_independent() {
        let spec = RelightSpec { subject_brighten: 1.0, ..RelightSpec::default() };
        let img = [0.3; 3];
        let out = apply(&img, 3, 1, 1, &[0.0, 0.5, 1.0], &[1.0; 3], &[face()], &spec).unwrap();
        assert_eq!(out[0], img[0]);
        assert!(((out[1] - img[1]) * 2.0 - (out[2] - img[2])).abs() < 1e-6);
        let moved = RelightSpec { light_x: 0.0, light_y: 1.0, light_size: 0.0, ..spec };
        assert_eq!(out, apply(&img, 3, 1, 1, &[0.0, 0.5, 1.0], &[0.0; 3], &[], &moved).unwrap());
    }

    #[test]
    fn nose_lifts_bridge_shades_sides_and_excludes_eyes_lips() {
        let img = vec![0.4; 101 * 101];
        let mask = vec![1.0; 101 * 101];
        let zeros = vec![0.0; 101 * 101];
        let spec = RelightSpec { nose_contour: 1.0, ..RelightSpec::default() };
        // Nose works without a person mask and independently of light controls.
        let out = apply(&img, 101, 101, 1, &zeros, &mask, &[face()], &spec).unwrap();
        let at = |x: usize, y: usize| out[y * 101 + x];
        assert!(at(50, 45) > 0.4);
        assert!(at(42, 45) < 0.4 && at(58, 45) < 0.4);
        for (x, y) in [(30, 30), (70, 30), (50, 73), (38, 73), (62, 73), (0, 0), (50, 90)] {
            assert_eq!(at(x, y).to_bits(), 0.4_f32.to_bits());
        }
        assert_eq!(apply(&img, 101, 101, 1, &mask, &zeros, &[face()], &spec).unwrap(), img);
        assert_eq!(apply(&img, 101, 101, 1, &mask, &mask, &[], &spec).unwrap(), img);
        // Duplicate detections must not double the effect.
        assert_eq!(out, apply(&img, 101, 101, 1, &zeros, &mask, &[face(), face()], &spec).unwrap());
    }

    #[test]
    fn nose_rotates_with_eye_axis_and_tip() {
        let img = vec![0.4; 101 * 101];
        let mask = vec![1.0; 101 * 101];
        let spec = RelightSpec { nose_contour: 1.0, ..RelightSpec::default() };
        let original = face();
        let mut turned = original.clone();
        let rotate = |p: Point| Point { x: 1.0 - p.y, y: p.x };
        turned.left_eye = rotate(original.left_eye);
        turned.right_eye = rotate(original.right_eye);
        turned.nose_tip = rotate(original.nose_tip);
        turned.mouth_left = rotate(original.mouth_left);
        turned.mouth_right = rotate(original.mouth_right);
        turned.mouth_center = rotate(original.mouth_center);
        let a = apply(&img, 101, 101, 1, &mask, &mask, &[original], &spec).unwrap();
        let b = apply(&img, 101, 101, 1, &mask, &mask, &[turned], &spec).unwrap();
        for y in 0..101 {
            for x in 0..101 {
                assert!((a[y * 101 + x] - b[x * 101 + (100 - y)]).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn invalid_parameters_and_masks_are_normalized() {
        let img = [0.3; 9];
        let mask = [1.0; 9];
        let invalid = RelightSpec {
            subject_brighten: f32::NAN,
            virtual_light: f32::INFINITY,
            nose_contour: f32::NEG_INFINITY,
            light_x: f32::NAN,
            light_y: f32::INFINITY,
            light_size: f32::NEG_INFINITY,
        };
        assert_eq!(apply(&img, 3, 3, 1, &mask, &mask, &[face()], &invalid).unwrap(), img);
        let bad_position = RelightSpec { virtual_light: 1.0, ..invalid };
        let centered = RelightSpec { virtual_light: 1.0, ..RelightSpec::default() };
        assert_eq!(apply(&img, 3, 3, 1, &mask, &mask, &[], &bad_position).unwrap(), apply(&img, 3, 3, 1, &mask, &mask, &[], &centered).unwrap());
        let clamped = RelightSpec { subject_brighten: 12.0, virtual_light: -2.0, nose_contour: -1.0, ..centered };
        let subject = RelightSpec { subject_brighten: 1.0, ..RelightSpec::default() };
        assert_eq!(apply(&img, 3, 3, 1, &mask, &mask, &[], &clamped).unwrap(), apply(&img, 3, 3, 1, &mask, &mask, &[], &subject).unwrap());
        let out = apply(&img[..3], 3, 1, 1, &[f32::NAN, f32::INFINITY, -1.0], &[0.0; 3], &[], &subject).unwrap();
        assert_eq!(out, img[..3]);
    }

    #[test]
    fn invalid_geometry_is_skipped_and_invalid_pixels_untouched() {
        let mut bad = face();
        bad.left_eye.x = f32::NAN;
        let mut degenerate = face();
        degenerate.right_eye = degenerate.left_eye;
        let img = vec![0.3; 25];
        let mask = vec![1.0; 25];
        let spec = RelightSpec { nose_contour: 1.0, ..RelightSpec::default() };
        assert_eq!(apply(&img, 5, 5, 1, &mask, &mask, &[bad, degenerate], &spec).unwrap(), img);
        let img = [f32::from_bits(0x7fc0_0123), 0.2, 0.1, -0.2, 0.4, 0.3, 1.2, 0.1, 0.2];
        let spec = RelightSpec { subject_brighten: 1.0, ..spec };
        assert_eq!(bits(&apply(&img, 3, 1, 3, &[1.0; 3], &[1.0; 3], &[], &spec).unwrap()), bits(&img));
    }

    #[test]
    fn dimensions_overflow_and_lengths_fail_even_for_identity() {
        let spec = RelightSpec::default();
        for (w, h, ch) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 5), (usize::MAX, 2, 1), (usize::MAX / 2, 1, 4), (usize::MAX / 8, 1, 1)] {
            assert!(apply(&[], w, h, ch, &[], &[], &[], &spec).is_err());
        }
        assert!(matches!(apply(&[], 1, 1, 1, &[1.0], &[1.0], &[], &spec), Err(Error::ShapeMismatch { .. })));
        assert!(matches!(apply(&[0.3], 1, 1, 1, &[], &[1.0], &[], &spec), Err(Error::ShapeMismatch { .. })));
        assert!(matches!(apply(&[0.3], 1, 1, 1, &[1.0], &[], &[], &spec), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn gray_matches_neutral_rgb_and_preserves_texture() {
        let img = [0.2, 0.6, 0.2, 0.6];
        let rgb: Vec<f32> = img.iter().flat_map(|v| [*v; 3]).collect();
        let spec = RelightSpec { subject_brighten: 0.7, virtual_light: 0.4, ..RelightSpec::default() };
        let gray = apply(&img, 2, 2, 1, &[1.0; 4], &[0.0; 4], &[], &spec).unwrap();
        let color = apply(&rgb, 2, 2, 3, &[1.0; 4], &[0.0; 4], &[], &spec).unwrap();
        for (g, c) in gray.iter().zip(color.chunks_exact(3)) {
            assert!((*g - c[0]).abs() < 1e-6);
            assert_eq!(c[0], c[1]);
            assert_eq!(c[1], c[2]);
        }
        assert!(gray[1] - gray[0] > 0.25);
        assert_eq!(gray[0], gray[2]);
        assert_eq!(gray[1], gray[3]);
    }
}
