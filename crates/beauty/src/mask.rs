//! Feathered mask math over plain `f32` buffers.
//!
//! Masks are row-major `width * height` slices in 0..1. All functions validate
//! geometry up front and return [`crate::Error`] instead of panicking.

use crate::Error;

fn check_mask(mask: &[f32], width: usize, height: usize, what: &'static str) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::EmptyInput(what));
    }
    let expected = width.checked_mul(height).ok_or_else(|| Error::InvalidParam(
        "mask dimensions overflow".to_string(),
    ))?;
    if mask.len() != expected {
        return Err(Error::ShapeMismatch {
            what,
            expected,
            found: mask.len(),
        });
    }
    Ok(())
}

/// Exact box blur (radius in pixels, clamped-edge) via integral image.
/// `radius == 0` returns a copy. Call twice to approximate a Gaussian.
pub fn feather_box(mask: &[f32], width: usize, height: usize, radius: usize) -> Result<Vec<f32>, Error> {
    check_mask(mask, width, height, "feather_box input")?;
    if radius == 0 {
        return Ok(mask.to_vec());
    }
    // Integral image with f64 accumulation for precision on large masks.
    let stride = width + 1;
    let mut integral = vec![0.0f64; stride * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0.0f64;
        for x in 0..width {
            let v = f64::from(mask[y * width + x].clamp(0.0, 1.0));
            row_sum += v;
            let idx = (y + 1) * stride + (x + 1);
            integral[idx] = integral[y * stride + (x + 1)] + row_sum;
        }
    }
    let r = radius as isize;
    let w = width as isize;
    let h = height as isize;
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        let y0 = (y as isize - r).max(0);
        let y1 = (y as isize + r).min(h - 1);
        for x in 0..width {
            let x0 = (x as isize - r).max(0);
            let x1 = (x as isize + r).min(w - 1);
            // Sum over [x0..=x1] x [y0..=y1], normalized by the *clamped*
            // window area so edges behave like clamped-edge extension.
            let a = integral[(y0 as usize) * stride + (x0 as usize)];
            let b = integral[(y0 as usize) * stride + (x1 as usize + 1)];
            let c = integral[(y1 as usize + 1) * stride + (x0 as usize)];
            let d = integral[(y1 as usize + 1) * stride + (x1 as usize + 1)];
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            out[y * width + x] = ((d - b - c + a) / area.max(1.0)) as f32;
        }
    }
    Ok(out)
}

/// Pixel-wise maximum of two masks.
pub fn union_max(a: &[f32], b: &[f32], width: usize, height: usize) -> Result<Vec<f32>, Error> {
    check_mask(a, width, height, "union_max a")?;
    check_mask(b, width, height, "union_max b")?;
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect())
}

/// Carves `cut` out of `base`: `base * (1 - cut)`, clamped to 0..1.
pub fn subtract(base: &[f32], cut: &[f32], width: usize, height: usize) -> Result<Vec<f32>, Error> {
    check_mask(base, width, height, "subtract base")?;
    check_mask(cut, width, height, "subtract cut")?;
    Ok(base
        .iter()
        .zip(cut.iter())
        .map(|(b, c)| (b * (1.0 - c)).clamp(0.0, 1.0))
        .collect())
}

/// Blends `processed` into `original` where `mask` is 1.
/// Buffers are interleaved `f32` with `channels` planes; `strength` scales the mask.
pub fn apply_masked_f32(
    original: &[f32],
    processed: &[f32],
    mask: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    strength: f32,
) -> Result<Vec<f32>, Error> {
    check_mask(mask, width, height, "apply_masked_f32 mask")?;
    if channels == 0 {
        return Err(Error::EmptyInput("apply_masked_f32 channels"));
    }
    let pixels = width.checked_mul(height).ok_or_else(|| Error::InvalidParam(
        "image dimensions overflow".to_string(),
    ))?;
    let expected = pixels.checked_mul(channels).ok_or_else(|| Error::InvalidParam(
        "image dimensions overflow".to_string(),
    ))?;
    if original.len() != expected {
        return Err(Error::ShapeMismatch {
            what: "apply_masked_f32 original",
            expected,
            found: original.len(),
        });
    }
    if processed.len() != expected {
        return Err(Error::ShapeMismatch {
            what: "apply_masked_f32 processed",
            expected,
            found: processed.len(),
        });
    }
    let s = if strength.is_finite() {
        strength.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if s <= 0.0 {
        return Ok(original.to_vec());
    }
    let mut out = Vec::with_capacity(expected);
    for (i, m) in mask.iter().enumerate() {
        let a = m.clamp(0.0, 1.0) * s;
        let base = i * channels;
        for c in 0..channels {
            let o = original[base + c];
            let p = processed[base + c];
            out.push(o + (p - o) * a);
        }
    }
    Ok(out)
}

/// 2D cross product (z-component) of OA x OB.
fn cross(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull (Andrew monotone chain), counter-clockwise, without duplicate endpoints.
/// Returns an empty vec for fewer than 3 distinct points.
pub fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut pts: Vec<[f32; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    if pts.len() < 3 {
        return Vec::new();
    }
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(core::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if pts.len() < 3 {
        return Vec::new();
    }
    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 {
            let n = lower.len();
            if cross(lower[n - 2], lower[n - 1], p) <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }
    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let n = upper.len();
            if cross(upper[n - 2], upper[n - 1], p) <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Scanline fill of a convex polygon into a float mask (1 inside, 0 outside).
pub fn fill_convex_poly(
    width: usize,
    height: usize,
    poly: &[[f32; 2]],
) -> Result<Vec<f32>, Error> {
    if width == 0 || height == 0 {
        return Err(Error::EmptyInput("fill_convex_poly dimensions"));
    }
    let pixels = width.checked_mul(height).ok_or_else(|| Error::InvalidParam(
        "mask dimensions overflow".to_string(),
    ))?;
    let mut out = vec![0.0f32; pixels];
    if poly.len() < 3 {
        return Ok(out);
    }
    // Intersect each scanline with polygon edges.
    for y in 0..height {
        let yc = y as f32 + 0.5;
        let mut xs: [f32; 64] = [0.0; 64];
        let mut n = 0usize;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[j];
            // Half-open rule on y to avoid double-counting vertices.
            let (lo, hi) = if a[1] < b[1] { (a, b) } else { (b, a) };
            if lo[1] <= yc && yc < hi[1] && n < xs.len() {
                let t = (yc - lo[1]) / (hi[1] - lo[1]).max(f32::MIN_POSITIVE);
                xs[n] = lo[0] + t * (hi[0] - lo[0]);
                n += 1;
            }
            j = i;
        }
        if n < 2 {
            continue;
        }
        // Tiny insertion sort; n is small for convex polygons.
        for i in 1..n {
            let mut k = i;
            while k > 0 && xs[k] < xs[k - 1] {
                xs.swap(k, k - 1);
                k -= 1;
            }
        }
        let mut k = 0;
        while k + 1 < n {
            let x0 = xs[k].max(0.0) as usize;
            let x1 = xs[k + 1].min(width as f32) as usize;
            for x in x0..x1.min(width) {
                out[y * width + x] = 1.0;
            }
            k += 2;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feather_zero_is_identity() {
        let m = vec![0.0f32, 0.5, 1.0, 0.25];
        assert_eq!(feather_box(&m, 2, 2, 0).unwrap(), m);
    }

    #[test]
    fn feather_spreads_impulse_and_preserves_mean() {
        let mut m = vec![0.0f32; 81];
        m[4 * 9 + 4] = 1.0;
        let out = feather_box(&m, 9, 9, 2).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
        assert!(out[4 * 9 + 4] < 1.0 && out[4 * 9 + 4] > 0.0);
        assert!(out[4 * 9 + 6] > 0.0);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn feather_rejects_bad_shapes() {
        assert!(feather_box(&[1.0; 3], 2, 2, 1).is_err());
        assert!(feather_box(&[], 0, 5, 1).is_err());
    }

    #[test]
    fn union_and_subtract() {
        let a = vec![1.0f32, 0.0, 0.5, 0.0];
        let b = vec![0.0f32, 1.0, 0.5, 0.0];
        assert_eq!(union_max(&a, &b, 2, 2).unwrap(), vec![1.0, 1.0, 0.5, 0.0]);
        let base = vec![1.0f32; 4];
        assert_eq!(subtract(&base, &a, 2, 2).unwrap(), vec![0.0, 1.0, 0.5, 1.0]);
        assert!(union_max(&a, &[1.0], 2, 2).is_err());
    }

    #[test]
    fn apply_masked_blends_inside_only() {
        let orig = vec![0.0f32; 2 * 2 * 3];
        let proc = vec![1.0f32; 2 * 2 * 3];
        let mask = vec![1.0f32, 0.0, 0.5, 0.0];
        let out = apply_masked_f32(&orig, &proc, &mask, 2, 2, 3, 1.0).unwrap();
        assert_eq!(out.len(), 12);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[3] - 0.0).abs() < 1e-6);
        assert!((out[6] - 0.5).abs() < 1e-6);
        let copy = apply_masked_f32(&orig, &proc, &mask, 2, 2, 3, 0.0).unwrap();
        assert_eq!(copy, orig);
        assert!(apply_masked_f32(&orig, &proc, &mask, 2, 2, 0, 1.0).is_err());
    }

    #[test]
    fn hull_of_square_has_four_corners() {
        let pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!(convex_hull(&[[0.0, 0.0], [1.0, 1.0]]).is_empty());
        assert!(convex_hull(&[[f32::NAN, 0.0], [0.0, 0.0], [1.0, 1.0]]).len() < 3);
    }

    #[test]
    fn fill_square_covers_interior() {
        let poly = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let m = fill_convex_poly(10, 10, &poly).unwrap();
        assert_eq!(m[4 * 10 + 4], 1.0);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[9 * 10 + 9], 0.0);
        let n: f32 = m.iter().sum();
        assert!((n - 16.0).abs() < 1.0, "filled={n}");
        assert!(fill_convex_poly(0, 5, &poly).is_err());
    }
}
