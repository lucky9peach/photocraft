//! Local light and detail: brighten, saturation, eye detail, sharpen.
//!
//! Same buffer conventions as [`crate::skin`]: interleaved `f32` 0..1,
//! alpha (when present) always preserved.

use photocraft_color::convert::rgb_to_gray;

use crate::Error;
use crate::mask::{apply_masked_f32, feather_box};

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

fn check_mask(mask: &[f32], w: usize, h: usize) -> Result<(), Error> {
    if mask.len() != w * h {
        return Err(Error::ShapeMismatch {
            what: "light mask",
            expected: w * h,
            found: mask.len(),
        });
    }
    Ok(())
}

/// Gamma lift inside the mask; 100% strength uses gamma 0.65.
pub fn brighten(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    face_mask: &[f32],
    strength: f32,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "brighten input")?;
    check_mask(face_mask, w, h)?;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 {
        return Ok(img.to_vec());
    }
    let gamma = 1.0 - 0.35 * s;
    let color_ch = if ch >= 3 { 3 } else { 1 };
    let mut out = img.to_vec();
    for i in 0..w * h {
        for c in 0..color_ch {
            let v = img[i * ch + c].clamp(0.0, 1.0);
            out[i * ch + c] = v.powf(gamma);
        }
    }
    apply_masked_f32(img, &out, face_mask, w, h, ch, 0.5 + 0.5 * s)
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 1e-9 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / d) + 2.0)
    } else {
        60.0 * (((r - g) / d) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max <= 1e-9 { 0.0 } else { d / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r1 + m, g1 + m, b1 + m]
}

/// Saturation adjust inside the mask: `amount` in -1..1, 0 = neutral.
/// Gray pixels are unaffected by construction.
pub fn adjust_saturation(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    face_mask: &[f32],
    amount: f32,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "adjust_saturation input")?;
    check_mask(face_mask, w, h)?;
    let a = if amount.is_finite() { amount.clamp(-1.0, 1.0) } else { 0.0 };
    if a == 0.0 || ch < 3 {
        return Ok(img.to_vec());
    }
    let mut out = img.to_vec();
    for i in 0..w * h {
        let (h, s, v) = rgb_to_hsv(img[i * ch], img[i * ch + 1], img[i * ch + 2]);
        let ns = if a > 0.0 { s * (1.0 + 0.6 * a) } else { s * (1.0 + 0.8 * a) };
        let rgb = hsv_to_rgb(h, ns.clamp(0.0, 1.0), v);
        out[i * ch] = rgb[0].clamp(0.0, 1.0);
        out[i * ch + 1] = rgb[1].clamp(0.0, 1.0);
        out[i * ch + 2] = rgb[2].clamp(0.0, 1.0);
    }
    apply_masked_f32(img, &out, face_mask, w, h, ch, (a.abs() * 2.0).min(1.0))
}

/// Compact CLAHE on a luma plane: 8x8 tiles, histogram clip, bilinear blend.
fn clahe(luma: &[f32], w: usize, h: usize, clip: f32) -> Vec<f32> {
    const TILES: usize = 8;
    const BINS: usize = 64;
    let tw = w.div_ceil(TILES).max(1);
    let th = h.div_ceil(TILES).max(1);
    // Per-tile clipped CDFs flattened as [ty * TILES + tx][bin].
    let mut cdfs = vec![vec![0.0f32; BINS]; TILES * TILES];
    for ty in 0..TILES {
        for tx in 0..TILES {
            let x0 = (tx * tw).min(w);
            let x1 = ((tx + 1) * tw).min(w).max(x0 + 1).min(w);
            let y0 = (ty * th).min(h);
            let y1 = ((ty + 1) * th).min(h).max(y0 + 1).min(h);
            let mut hist = [0usize; BINS];
            let mut n = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let b = (luma[y * w + x].clamp(0.0, 1.0) * (BINS - 1) as f32) as usize;
                    hist[b.min(BINS - 1)] += 1;
                    n += 1;
                }
            }
            let limit = ((clip * n as f32 / BINS as f32).max(1.0)) as usize;
            let mut excess = 0usize;
            for hbin in hist.iter_mut() {
                if *hbin > limit {
                    excess += *hbin - limit;
                    *hbin = limit;
                }
            }
            let per = excess / BINS;
            let mut rem = excess % BINS;
            for hbin in hist.iter_mut() {
                *hbin += per;
                if rem > 0 {
                    *hbin += 1;
                    rem -= 1;
                }
            }
            let total = (n + excess) as f32;
            let mut acc = 0.0f32;
            let cdf = &mut cdfs[ty * TILES + tx];
            for (b, slot) in hist.iter().zip(cdf.iter_mut()) {
                acc += *b as f32 / total;
                *slot = acc;
            }
        }
    }
    let sample = |tx: usize, ty: usize, v: f32| -> f32 {
        let tx = tx.min(TILES - 1);
        let ty = ty.min(TILES - 1);
        let cdf = &cdfs[ty * TILES + tx];
        let pos = v.clamp(0.0, 1.0) * (BINS - 1) as f32;
        let b0 = (pos.floor() as usize).min(BINS - 1);
        let b1 = (b0 + 1).min(BINS - 1);
        let f = pos - b0 as f32;
        cdf[b0] * (1.0 - f) + cdf[b1] * f
    };
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = luma[y * w + x].clamp(0.0, 1.0);
            // Bilinear blend between the four surrounding tile CDFs.
            let gx = (x as f32 / tw as f32 - 0.5).max(0.0);
            let gy = (y as f32 / th as f32 - 0.5).max(0.0);
            let tx0 = (gx.floor() as usize).min(TILES - 1);
            let ty0 = (gy.floor() as usize).min(TILES - 1);
            let tx1 = (tx0 + 1).min(TILES - 1);
            let ty1 = (ty0 + 1).min(TILES - 1);
            let fx = (gx - tx0 as f32).clamp(0.0, 1.0);
            let fy = (gy - ty0 as f32).clamp(0.0, 1.0);
            let v00 = sample(tx0, ty0, v);
            let v10 = sample(tx1, ty0, v);
            let v01 = sample(tx0, ty1, v);
            let v11 = sample(tx1, ty1, v);
            out[y * w + x] = (v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy)
                .clamp(0.0, 1.0);
        }
    }
    out
}

/// Eye-region detail: CLAHE on luma + gentle lift, chroma-preserving
/// (RGB scaled by the luma ratio), blended through the eye mask.
pub fn enhance_eye_detail(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    eye_mask: &[f32],
    strength: f32,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "enhance_eye_detail input")?;
    check_mask(eye_mask, w, h)?;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 || ch < 3 {
        return Ok(img.to_vec());
    }
    let pixels = w * h;
    let mut luma = vec![0.0f32; pixels];
    for i in 0..pixels {
        luma[i] = rgb_to_gray([img[i * ch], img[i * ch + 1], img[i * ch + 2]]).clamp(0.0, 1.0);
    }
    let eq = clahe(&luma, w, h, 2.0 + 2.0 * s);
    let mut out = img.to_vec();
    for i in 0..pixels {
        let lifted = (eq[i] + 0.03 * s).clamp(0.0, 1.0);
        let ratio = if luma[i] > 1e-4 { lifted / luma[i] } else { 1.0 };
        let ratio = ratio.clamp(0.5, 2.5);
        for c in 0..3 {
            out[i * ch + c] = (img[i * ch + c] * ratio).clamp(0.0, 1.0);
        }
    }
    apply_masked_f32(img, &out, eye_mask, w, h, ch, 0.4 + 0.6 * s)
}

/// Unsharp-mask sharpening inside the mask.
pub fn sharpen(
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
    face_mask: &[f32],
    strength: f32,
    radius: usize,
) -> Result<Vec<f32>, Error> {
    check_image(img, w, h, ch, "sharpen input")?;
    check_mask(face_mask, w, h)?;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    if s <= 0.0 {
        return Ok(img.to_vec());
    }
    let color_ch = if ch >= 3 { 3 } else { 1 };
    let mut sharp = img.to_vec();
    for c in 0..color_ch {
        let mut plane = Vec::with_capacity(w * h);
        for i in 0..w * h {
            plane.push(img[i * ch + c]);
        }
        let blur = feather_box(&plane, w, h, radius.max(1))?;
        for i in 0..w * h {
            sharp[i * ch + c] = plane[i] * (1.0 + 0.8 * s) - blur[i] * (0.8 * s);
        }
    }
    apply_masked_f32(img, &sharp, face_mask, w, h, ch, 0.5 + 0.5 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, ch: usize, v: f32) -> Vec<f32> {
        vec![v; w * h * ch]
    }

    #[test]
    fn zeros_are_identity() {
        let img = flat(8, 8, 3, 0.5);
        let mask = vec![1.0f32; 64];
        assert_eq!(brighten(&img, 8, 8, 3, &mask, 0.0).unwrap(), img);
        assert_eq!(adjust_saturation(&img, 8, 8, 3, &mask, 0.0).unwrap(), img);
        assert_eq!(enhance_eye_detail(&img, 8, 8, 3, &mask, 0.0).unwrap(), img);
        assert_eq!(sharpen(&img, 8, 8, 3, &mask, 0.0, 2).unwrap(), img);
    }

    #[test]
    fn brighten_lifts_masked_only() {
        let img = flat(8, 8, 3, 0.4);
        let mut mask = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..4 {
                mask[y * 8 + x] = 1.0;
            }
        }
        let out = brighten(&img, 8, 8, 3, &mask, 1.0).unwrap();
        assert!(out[0] > 0.5, "masked half lifted");
        assert_eq!(out[4 * 3], img[4 * 3], "unmasked bit-identical");
    }

    #[test]
    fn saturation_ignores_gray_and_boosts_color() {
        let gray = flat(4, 4, 3, 0.5);
        let mask = vec![1.0f32; 16];
        let out = adjust_saturation(&gray, 4, 4, 3, &mask, 1.0).unwrap();
        for v in out {
            assert!((v - 0.5).abs() < 1e-5, "gray must not shift hue");
        }
        let mut color = flat(4, 4, 3, 0.5);
        for i in 0..16 {
            color[i * 3] = 0.8;
        }
        let out = adjust_saturation(&color, 4, 4, 3, &mask, 1.0).unwrap();
        let (_, s0, _) = rgb_to_hsv(0.8, 0.5, 0.5);
        let (_, s1, _) = rgb_to_hsv(out[0], out[1], out[2]);
        assert!(s1 > s0, "saturation should increase");
        let hsv = rgb_to_hsv(0.2, 0.5, 0.9);
        let back = hsv_to_rgb(hsv.0, hsv.1, hsv.2);
        assert!((back[0] - 0.2).abs() + (back[1] - 0.5).abs() + (back[2] - 0.9).abs() < 1e-4);
    }

    #[test]
    fn eye_detail_adds_local_contrast() {
        // Smooth ramp + faint ripple: enhancement must amplify the ripple
        // (high-pass energy), which is what "detail" means for CLAHE.
        let (w, h) = (64, 64);
        let mut img = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = 0.35
                    + 0.3 * (x as f32 / w as f32)
                    + 0.015 * ((x as f32 * 1.7).sin() + (y as f32 * 2.3).cos());
                for c in 0..3 {
                    img[(y * w + x) * 3 + c] = v;
                }
            }
        }
        let mask = vec![1.0f32; w * h];
        let out = enhance_eye_detail(&img, w, h, 3, &mask, 1.0).unwrap();
        let energy = |v: &[f32]| {
            let luma: Vec<f32> = (0..w * h).map(|i| v[i * 3]).collect();
            let bg = feather_box(&luma, w, h, 3).unwrap();
            let e: f32 = luma.iter().zip(bg.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                / (w * h) as f32;
            e
        };
        let (before, after) = (energy(&img), energy(&out));
        assert!(after > before * 1.5, "detail {before} -> {after}");
    }

    #[test]
    fn sharpen_steepens_edge() {
        let mut img = vec![0.0f32; 16 * 8 * 3];
        for y in 0..8 {
            for x in 0..16 {
                let v = if x < 8 { 0.3 } else { 0.7 };
                for c in 0..3 {
                    img[(y * 16 + x) * 3 + c] = v;
                }
            }
        }
        let mask = vec![1.0f32; 16 * 8];
        let out = sharpen(&img, 16, 8, 3, &mask, 1.0, 2).unwrap();
        // Overshoot: dark side dips below 0.3 near the edge, bright exceeds 0.7.
        assert!(out[(4 * 16 + 7) * 3] < 0.3, "dark-side overshoot");
        assert!(out[(4 * 16 + 8) * 3] > 0.7, "bright-side overshoot");
    }
}
