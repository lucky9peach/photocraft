//! Skin processing: edge-preserving smoothing, tone uniform, blemish fade.
//!
//! Buffers are interleaved `f32` in 0..1, row-major, with `channels` planes.
//! Alpha (when present) is preserved structurally: only color planes change.
//! RGB pipelines use the first three channels; single-channel (gray) input
//! supports smoothing/brightening-style ops with chroma steps skipped.
//!
//! Lab work uses [`photocraft_color`] sRGB converters; the engine integration
//! (Stage 6) is responsible for feeding working-space RGB.

use photocraft_color::convert::{lab_to_srgb, srgb_to_lab};

use crate::Error;
use crate::mask::{apply_masked_f32, feather_box};

/// Box blur approximating a gaussian: three passes ≈ σ with `radius ≈ σ`.
fn gauss_approx(channel: &[f32], w: usize, h: usize, radius: usize) -> Result<Vec<f32>, Error> {
    let once = feather_box(channel, w, h, radius)?;
    let twice = feather_box(&once, w, h, radius)?;
    feather_box(&twice, w, h, radius)
}

fn check_image(img: &[f32], w: usize, h: usize, ch: usize, what: &'static str) -> Result<(), Error> {
    if w == 0 || h == 0 || ch == 0 {
        return Err(Error::EmptyInput(what));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam(
        "image dimensions overflow".to_string(),
    ))?;
    let expected = pixels.checked_mul(ch).ok_or_else(|| Error::InvalidParam(
        "image dimensions overflow".to_string(),
    ))?;
    if img.len() != expected {
        return Err(Error::ShapeMismatch {
            what,
            expected,
            found: img.len(),
        });
    }
    Ok(())
}

fn plane(img: &[f32], w: usize, h: usize, ch: usize, c: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for i in 0..w * h {
        out.push(img[i * ch + c]);
    }
    out
}

fn put_plane(img: &mut [f32], w: usize, h: usize, ch: usize, c: usize, plane: &[f32]) {
    for i in 0..w * h {
        img[i * ch + c] = plane[i];
    }
}

/// Guided filter (He et al. 2010), O(1) per pixel: smooths `src` while
/// following edges of `guide`. Both are single-channel `w*h` buffers.
pub fn guided_filter(
    guide: &[f32],
    src: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    eps: f32,
) -> Result<Vec<f32>, Error> {
    if guide.len() != w * h || src.len() != w * h {
        return Err(Error::ShapeMismatch {
            what: "guided_filter input",
            expected: w * h,
            found: guide.len().min(src.len()),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error::InvalidParam("guided eps must be finite >= 0".into()));
    }
    let mean_i = feather_box(guide, w, h, radius)?;
    let mean_p = feather_box(src, w, h, radius)?;
    let corr_i: Vec<f32> = guide.iter().map(|v| v * v).collect();
    let corr_i = feather_box(&corr_i, w, h, radius)?;
    let corr_ip: Vec<f32> = guide.iter().zip(src.iter()).map(|(g, p)| g * p).collect();
    let corr_ip = feather_box(&corr_ip, w, h, radius)?;
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for i in 0..w * h {
        let var_i = (corr_i[i] - mean_i[i] * mean_i[i]).max(0.0);
        let cov = corr_ip[i] - mean_i[i] * mean_p[i];
        let ai = cov / (var_i + eps);
        a[i] = ai;
        b[i] = mean_p[i] - ai * mean_i[i];
    }
    let mean_a = feather_box(&a, w, h, radius)?;
    let mean_b = feather_box(&b, w, h, radius)?;
    Ok(guide
        .iter()
        .zip(mean_a.iter())
        .zip(mean_b.iter())
        .map(|((g, ma), mb)| ma * g + mb)
        .collect())
}

/// Edge-preserving skin smoothing inside `skin_mask`.
///
/// Each RGB channel is self-guided-filtered, then `texture_keep` (0..1) of the
/// original detail is restored to avoid a plastic look, and the result is
/// blended by `strength` (0..1) through the mask. `strength <= 0` is exact copy.
/// Alpha (when present) is preserved structurally: only color planes change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothSpec {
    pub strength: f32,
    pub texture_keep: f32,
    pub radius: usize,
    pub eps: f32,
}

pub fn smooth_skin(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    skin_mask: &[f32],
    spec: &SmoothSpec,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "smooth_skin input")?;
    let strength = spec.strength;
    let texture_keep = spec.texture_keep;
    let radius = spec.radius;
    let eps = spec.eps;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 {
        return Ok(img.to_vec());
    }
    let keep = if texture_keep.is_finite() {
        texture_keep.clamp(0.0, 1.0)
    } else {
        0.7
    };
    let detail_w = keep * 0.5;
    let color_ch = if ch >= 3 { 3 } else { 1 };
    let mut smoothed = img.to_vec();
    for c in 0..color_ch {
        let src = plane(img, w, h, ch, c);
        let sm = guided_filter(&src, &src, w, h, radius.max(1), eps)?;
        let mut blended = vec![0.0f32; w * h];
        for i in 0..w * h {
            blended[i] = sm[i] * (1.0 - detail_w) + src[i] * detail_w;
        }
        put_plane(&mut smoothed, w, h, ch, c, &blended);
    }
    apply_masked_f32(img, &smoothed, skin_mask, w, h, ch, s)
}

/// Pulls Lab a/b channels toward their local mean inside the mask (RGB only).
pub fn uniform_tone(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    skin_mask: &[f32],
    strength: f32,
    radius: usize,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "uniform_tone input")?;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 || ch < 3 {
        return Ok(img.to_vec());
    }
    let pixels = w * h;
    let mut l = vec![0.0f32; pixels];
    let mut a = vec![0.0f32; pixels];
    let mut b = vec![0.0f32; pixels];
    for i in 0..pixels {
        let lab = srgb_to_lab([img[i * ch], img[i * ch + 1], img[i * ch + 2]]);
        l[i] = lab[0];
        a[i] = lab[1];
        b[i] = lab[2];
    }
    let mean_a = feather_box(&a, w, h, radius.max(1))?;
    let mean_b = feather_box(&b, w, h, radius.max(1))?;
    let mut out = img.to_vec();
    for i in 0..pixels {
        let na = a[i] * (1.0 - 0.6 * s) + mean_a[i] * (0.6 * s);
        let nb = b[i] * (1.0 - 0.6 * s) + mean_b[i] * (0.6 * s);
        let rgb = lab_to_srgb([l[i], na, nb]);
        out[i * ch] = rgb[0].clamp(0.0, 1.0);
        out[i * ch + 1] = rgb[1].clamp(0.0, 1.0);
        out[i * ch + 2] = rgb[2].clamp(0.0, 1.0);
    }
    apply_masked_f32(img, &out, skin_mask, w, h, ch, 0.35 + 0.65 * s)
}

/// 3x3 minimum filter with clamped edges.
fn min3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::INFINITY;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    m = m.min(src[yy * w + xx]);
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

/// 3x3 maximum filter with clamped edges.
fn max3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    m = m.max(src[yy * w + xx]);
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

/// Fades small high-contrast spots: dual-scale box background, threshold,
/// 3x3 open + dilate, then blends a blurred copy only on the spots.
pub fn fade_blemishes(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    skin_mask: &[f32],
    strength: f32,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "fade_blemishes input")?;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 {
        return Ok(img.to_vec());
    }
    let color_ch = if ch >= 3 { 3 } else { 1 };
    // Max over channels of |img - background| at two scales.
    let mut diff = vec![0.0f32; w * h];
    for c in 0..color_ch {
        let src = plane(img, w, h, ch, c);
        let bg7 = feather_box(&src, w, h, 7)?;
        let bg15 = feather_box(&src, w, h, 15)?;
        for i in 0..w * h {
            let d = (src[i] - bg7[i]).abs().max((src[i] - bg15[i]).abs());
            diff[i] = diff[i].max(d);
        }
    }
    let thresh = (30.0 - 18.0 * s) / 255.0;
    let spots: Vec<f32> = diff.iter().map(|d| if *d > thresh { 1.0 } else { 0.0 }).collect();
    let opened = max3(&min3(&spots, w, h), w, h);
    let dilated = max3(&opened, w, h);
    let feathered = feather_box(&dilated, w, h, 2)?;
    if skin_mask.len() != w * h {
        return Err(Error::ShapeMismatch {
            what: "fade_blemishes mask",
            expected: w * h,
            found: skin_mask.len(),
        });
    }
    let spot_mask: Vec<f32> = feathered
        .iter()
        .zip(skin_mask.iter())
        .map(|(a, b)| (a * b).clamp(0.0, 1.0))
        .collect();
    if spot_mask.iter().fold(0.0f32, |m, v| m.max(*v)) <= 0.0 {
        return Ok(img.to_vec());
    }
    let mut blurred = img.to_vec();
    for c in 0..color_ch {
        let src = plane(img, w, h, ch, c);
        let sm = gauss_approx(&src, w, h, 2)?;
        put_plane(&mut blurred, w, h, ch, c, &sm);
    }
    apply_masked_f32(img, &blurred, &spot_mask, w, h, ch, 0.4 + 0.6 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, ch: usize, v: f32) -> Vec<f32> {
        vec![v; w * h * ch]
    }

    #[test]
    fn guided_flat_is_identity() {
        let g = vec![0.5f32; 64];
        let out = guided_filter(&g, &g, 8, 8, 2, 1e-4).unwrap();
        for v in out {
            assert!((v - 0.5).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn guided_preserves_step_edge() {
        // Left half 0.2, right half 0.8: heavy smoothing must keep the edge.
        let mut g = vec![0.0f32; 12 * 12];
        for y in 0..12 {
            for x in 0..12 {
                g[y * 12 + x] = if x < 6 { 0.2 } else { 0.8 };
            }
        }
        let out = guided_filter(&g, &g, 12, 12, 3, 1e-6).unwrap();
        assert!((out[6 * 12 + 2] - 0.2).abs() < 0.02);
        assert!((out[6 * 12 + 9] - 0.8).abs() < 0.02);
        // Edge transition stays sharp: adjacent columns differ strongly.
        assert!((out[6 * 12 + 6] - out[6 * 12 + 5]).abs() > 0.3);
    }

    #[test]
    fn smooth_reduces_noise_and_zero_is_copy() {
        // Deterministic pseudo-noise.
        let mut img = flat(24, 24, 3, 0.7);
        let mut x = 0x1234_5678u32;
        for v in img.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *v += ((x >> 8) as f32 / 16_777_216.0 - 0.5) * 0.1;
        }
        let mask = vec![1.0f32; 24 * 24];
        let spec = SmoothSpec { strength: 0.0, texture_keep: 0.7, radius: 2, eps: 1e-4 };
        let zero = smooth_skin(&img, 24, 24, 3, &mask, &spec).unwrap();
        assert_eq!(zero, img);
        let spec = SmoothSpec { strength: 1.0, texture_keep: 0.2, radius: 2, eps: 1e-4 };
        let out = smooth_skin(&img, 24, 24, 3, &mask, &spec).unwrap();
        let var = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
        };
        assert!(var(&out) < var(&img) * 0.9, "noise not reduced");
    }

    #[test]
    fn smooth_preserves_alpha() {
        let mut img = flat(8, 8, 4, 0.6);
        for i in 0..64 {
            img[i * 4 + 3] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let mask = vec![1.0f32; 64];
        let spec = SmoothSpec { strength: 1.0, texture_keep: 0.0, radius: 2, eps: 1e-4 };
        let out = smooth_skin(&img, 8, 8, 4, &mask, &spec).unwrap();
        for i in 0..64 {
            assert_eq!(out[i * 4 + 3], img[i * 4 + 3]);
        }
    }

    #[test]
    fn tone_pulls_chroma_together() {
        // Two halves with different hue; tone uniform must reduce ab spread.
        let mut img = vec![0.0f32; 16 * 8 * 3];
        for y in 0..8 {
            for x in 0..16 {
                let c = if x < 8 { [0.75, 0.55, 0.5] } else { [0.55, 0.6, 0.55] };
                img[(y * 16 + x) * 3..(y * 16 + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        let mask = vec![1.0f32; 16 * 8];
        let spread = |v: &[f32]| {
            let labs: Vec<[f32; 3]> = (0..16 * 8)
                .map(|i| srgb_to_lab([v[i * 3], v[i * 3 + 1], v[i * 3 + 2]]))
                .collect();
            let ma = labs.iter().map(|l| l[1]).sum::<f32>() / labs.len() as f32;
            let mb = labs.iter().map(|l| l[2]).sum::<f32>() / labs.len() as f32;
            labs.iter().map(|l| (l[1] - ma).abs() + (l[2] - mb).abs()).sum::<f32>()
        };
        let before = spread(&img);
        let out = uniform_tone(&img, 16, 8, 3, &mask, 1.0, 4).unwrap();
        assert!(spread(&out) < before, "chroma spread not reduced");
        let zero = uniform_tone(&img, 16, 8, 3, &mask, 0.0, 4).unwrap();
        assert_eq!(zero, img);
    }

    #[test]
    fn blemish_fades_dark_spot() {
        let mut img = flat(32, 32, 3, 0.7);
        for y in 14..20 {
            for x in 14..20 {
                for c in 0..3 {
                    img[(y * 32 + x) * 3 + c] = 0.15;
                }
            }
        }
        let mask = vec![1.0f32; 32 * 32];
        let out = fade_blemishes(&img, 32, 32, 3, &mask, 0.9).unwrap();
        let center = out[(17 * 32 + 17) * 3];
        assert!(center > 0.35, "spot not faded: {center}");
        // Far corner untouched.
        assert!((out[0] - 0.7).abs() < 1e-6);
        let zero = fade_blemishes(&img, 32, 32, 3, &mask, 0.0).unwrap();
        assert_eq!(zero, img);
    }
}
