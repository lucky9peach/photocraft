//! Unified displacement-field warp engine for face + body reshaping.
//!
//! Every warp adds a smooth (cosine, C1-continuous) term to one field, and the
//! field is applied with a single bilinear remap: `out(p) = src(p + d(p))`.
//! Because the field stores *sampling* offsets, moving content along `+v`
//! means sampling from `-v`; the primitives take intuitive *content* directions
//! and negate internally (regression-tested below).
//!
//! Primitives: [`Field::add_radial_scale`] (eye enlarge), [`Field::add_pull`]
//! (jaw slim, mouth-corner lift), [`Field::add_axis_squeeze`] (nose/waist
//! narrowing), [`Field::add_segment_stretch`] (leg lengthening). Each accepts
//! an optional person/face mask so warps never touch the background.

use crate::Error;

/// Hard cap on field pixels (hostile-input guard).
const MAX_PIXELS: usize = 200_000_000;

/// Cosine falloff: 1 at distance 0, 0 at `radius`, C1-continuous.
fn falloff(dist: f32, radius: f32) -> f32 {
    if !radius.is_finite() || radius <= 0.0 || !dist.is_finite() {
        return 0.0;
    }
    if dist >= radius {
        return 0.0;
    }
    let t = (dist / radius).clamp(0.0, 1.0);
    0.5 + 0.5 * (core::f32::consts::PI * t).cos()
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn check_field_dims(w: usize, h: usize) -> Result<usize, Error> {
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("warp field dimensions"));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam(
        "warp field dimensions overflow".to_string(),
    ))?;
    if pixels > MAX_PIXELS {
        return Err(Error::InvalidParam(format!(
            "warp field too large: {pixels} pixels"
        )));
    }
    Ok(pixels)
}

/// Sampling-offset displacement field over a `w*h` pixel grid.
#[derive(Clone, Debug)]
pub struct Field {
    w: usize,
    h: usize,
    dx: Vec<f32>,
    dy: Vec<f32>,
}

/// Directional pull: content moves along `(vx, vy)` (need not be unit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PullSpec {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub amount: f32,
    pub radius: f32,
}

/// Elliptical squeeze about a centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqueezeSpec {
    pub cx: f32,
    pub cy: f32,
    pub rx: f32,
    pub ry: f32,
    pub strength: f32,
    pub horizontal: bool,
}

/// Limb-segment stretch from joint A to joint B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StretchSpec {
    pub ax: f32,
    pub ay: f32,
    pub bx: f32,
    pub by: f32,
    pub width: f32,
    pub extend: f32,
}

impl Field {
    pub fn new(w: usize, h: usize) -> Result<Self, Error> {
        let pixels = check_field_dims(w, h)?;
        Ok(Self {
            w,
            h,
            dx: vec![0.0; pixels],
            dy: vec![0.0; pixels],
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn dx(&self) -> &[f32] {
        &self.dx
    }

    pub fn dy(&self) -> &[f32] {
        &self.dy
    }

    /// Maximum absolute offset anywhere in the field.
    pub fn max_abs(&self) -> f32 {
        self.dx
            .iter()
            .chain(self.dy.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()))
    }

    fn check_mask(&self, mask: Option<&[f32]>) -> Result<(), Error> {
        if let Some(m) = mask {
            let expected = self.dx.len();
            if m.len() != expected {
                return Err(Error::ShapeMismatch {
                    what: "warp gate mask",
                    expected,
                    found: m.len(),
                });
            }
        }
        Ok(())
    }

    fn gate(&self, mask: Option<&[f32]>, idx: usize) -> f32 {
        match mask {
            None => 1.0,
            Some(m) => m.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        }
    }

    /// Radial scale about `(cx, cy)`: `strength` in 0..1 enlarges up to ~18%.
    /// `mask` (same dims, 0..1) optionally gates the effect per pixel.
    pub fn add_radial_scale(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        strength: f32,
        mask: Option<&[f32]>,
    ) -> Result<(), Error> {
        if ![cx, cy, radius, strength].iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidParam("radial scale args must be finite".into()));
        }
        self.check_mask(mask)?;
        if strength <= 0.0 || radius <= 2.0 {
            return Ok(());
        }
        let k = 0.18 * strength.clamp(0.0, 1.0);
        let x0 = (cx - radius).max(0.0) as usize;
        let x1 = ((cx + radius).min(self.w as f32).ceil() as usize).min(self.w);
        let y0 = (cy - radius).max(0.0) as usize;
        let y1 = ((cy + radius).min(self.h as f32).ceil() as usize).min(self.h);
        for y in y0..y1 {
            for x in x0..x1 {
                let ox = x as f32 + 0.5 - cx;
                let oy = y as f32 + 0.5 - cy;
                let dist = ox.hypot(oy);
                let f = falloff(dist, radius);
                if f > 0.0 {
                    let idx = y * self.w + x;
                    let g = self.gate(mask, idx) * f * k;
                    // Sample closer to the centre => content expands outward.
                    self.dx[idx] += -ox * g;
                    self.dy[idx] += -oy * g;
                }
            }
        }
        Ok(())
    }

    /// Pulls content along unit-ish direction `(vx, vy)` by up to `amount` px.
    /// `dir` is the *content* motion; sampling offsets point the other way.
    pub fn add_pull(&mut self, spec: &PullSpec, mask: Option<&[f32]>) -> Result<(), Error> {
        let PullSpec { x: px, y: py, vx, vy, amount, radius } = *spec;
        if ![px, py, vx, vy, amount, radius].iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidParam("pull args must be finite".into()));
        }
        self.check_mask(mask)?;
        let len = vx.hypot(vy);
        if amount <= 0.0 || radius <= 2.0 || len < 1e-6 {
            return Ok(());
        }
        let (ux, uy) = (vx / len, vy / len);
        let x0 = (px - radius).max(0.0) as usize;
        let x1 = ((px + radius).min(self.w as f32).ceil() as usize).min(self.w);
        let y0 = (py - radius).max(0.0) as usize;
        let y1 = ((py + radius).min(self.h as f32).ceil() as usize).min(self.h);
        for y in y0..y1 {
            for x in x0..x1 {
                let dist = (x as f32 + 0.5 - px).hypot(y as f32 + 0.5 - py);
                let f = falloff(dist, radius);
                if f > 0.0 {
                    let idx = y * self.w + x;
                    let g = self.gate(mask, idx) * f * amount;
                    // Sample from behind the motion => content moves along +v.
                    self.dx[idx] += -ux * g;
                    self.dy[idx] += -uy * g;
                }
            }
        }
        Ok(())
    }

    /// Elliptical squeeze about `(cx, cy)`: `strength` in 0..1 compresses
    /// horizontally (or vertically) up to ~10% at the centre.
    pub fn add_axis_squeeze(&mut self, spec: &SqueezeSpec, mask: Option<&[f32]>) -> Result<(), Error> {
        let SqueezeSpec { cx, cy, rx, ry, strength, horizontal } = *spec;
        if ![cx, cy, rx, ry, strength].iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidParam("squeeze args must be finite".into()));
        }
        self.check_mask(mask)?;
        if strength <= 0.0 || rx <= 1.0 || ry <= 1.0 {
            return Ok(());
        }
        let k = 0.10 * strength.clamp(0.0, 1.0);
        let x0 = (cx - rx).max(0.0) as usize;
        let x1 = ((cx + rx).min(self.w as f32).ceil() as usize).min(self.w);
        let y0 = (cy - ry).max(0.0) as usize;
        let y1 = ((cy + ry).min(self.h as f32).ceil() as usize).min(self.h);
        for y in y0..y1 {
            for x in x0..x1 {
                let nx = (x as f32 + 0.5 - cx) / rx;
                let ny = (y as f32 + 0.5 - cy) / ry;
                let f = falloff(nx.hypot(ny), 1.0);
                if f > 0.0 {
                    let idx = y * self.w + x;
                    let g = self.gate(mask, idx) * f * k;
                    // Sample outward from the axis => content narrows.
                    if horizontal {
                        self.dx[idx] += (x as f32 + 0.5 - cx) * g;
                    } else {
                        self.dy[idx] += (y as f32 + 0.5 - cy) * g;
                    }
                }
            }
        }
        Ok(())
    }

    /// Lengthens a limb segment from joint A (`ax`, `ay`) to joint B (`bx`, `by`)
    /// by `extend` fraction of its length (0..0.5). Content at/below B moves
    /// along A→B; lateral falloff spans `width`. Gate with the person mask so
    /// the background stays pixel-identical.
    pub fn add_segment_stretch(&mut self, spec: &StretchSpec, mask: Option<&[f32]>) -> Result<(), Error> {
        let StretchSpec { ax, ay, bx, by, width, extend } = *spec;
        if ![ax, ay, bx, by, width, extend].iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidParam("stretch args must be finite".into()));
        }
        self.check_mask(mask)?;
        let abx = bx - ax;
        let aby = by - ay;
        let len = abx.hypot(aby);
        if extend <= 0.0 || width <= 2.0 || len < 1e-3 {
            return Ok(());
        }
        let e = extend.clamp(0.0, 0.5);
        let (ux, uy) = (abx / len, aby / len);
        // Bound the affected region generously; per-pixel falloff does the shaping.
        let pad = width + e * len + 2.0;
        let x0 = (ax.min(bx) - pad).max(0.0) as usize;
        let x1 = ((ax.max(bx) + pad).min(self.w as f32).ceil() as usize).min(self.w);
        let y0 = (ay.min(by) - pad).max(0.0) as usize;
        let y1 = ((ay.max(by) + pad).min(self.h as f32).ceil() as usize).min(self.h);
        let blend = (0.3 * len).max(4.0);
        for y in y0..y1 {
            for x in x0..x1 {
                let rx = x as f32 + 0.5 - ax;
                let ry = y as f32 + 0.5 - ay;
                let s = rx * ux + ry * uy;
                let lat = (rx - s * ux).hypot(ry - s * uy);
                let lateral = falloff(lat, width);
                if lateral <= 0.0 {
                    continue;
                }
                let axial = if s < 0.0 {
                    0.0
                } else if s <= len {
                    smoothstep(s / len)
                } else {
                    1.0 - smoothstep((s - len) / blend)
                };
                if axial <= 0.0 {
                    continue;
                }
                let idx = y * self.w + x;
                let g = self.gate(mask, idx) * lateral * axial * e * len;
                // Content moves along +u (B shifts outward); sample from -u.
                self.dx[idx] += -ux * g;
                self.dy[idx] += -uy * g;
            }
        }
        Ok(())
    }

    /// Applies the field to interleaved `f32` samples with clamped-edge
    /// bilinear sampling. All channels (incl. alpha) warp uniformly.
    pub fn remap(&self, src: &[f32], channels: usize) -> Result<Vec<f32>, Error> {
        if channels == 0 {
            return Err(Error::EmptyInput("remap channels"));
        }
        let pixels = self.dx.len();
        let expected = pixels.checked_mul(channels).ok_or_else(|| Error::InvalidParam(
            "remap dimensions overflow".to_string(),
        ))?;
        if src.len() != expected {
            return Err(Error::ShapeMismatch {
                what: "remap src",
                expected,
                found: src.len(),
            });
        }
        if self.max_abs() == 0.0 {
            return Ok(src.to_vec());
        }
        let w = self.w as f32;
        let h = self.h as f32;
        let mut out = vec![0.0f32; expected];
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let sx = (x as f32 + self.dx[idx]).clamp(0.0, w - 1.0);
                let sy = (y as f32 + self.dy[idx]).clamp(0.0, h - 1.0);
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(self.w - 1);
                let y1 = (y0 + 1).min(self.h - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                let dst = idx * channels;
                for c in 0..channels {
                    let p00 = src[(y0 * self.w + x0) * channels + c];
                    let p10 = src[(y0 * self.w + x1) * channels + c];
                    let p01 = src[(y1 * self.w + x0) * channels + c];
                    let p11 = src[(y1 * self.w + x1) * channels + c];
                    out[dst + c] = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_field_roundtrips() {
        let f = Field::new(8, 6).unwrap();
        assert_eq!(f.max_abs(), 0.0);
        let src: Vec<f32> = (0..8 * 6 * 3).map(|i| i as f32 / 100.0).collect();
        assert_eq!(f.remap(&src, 3).unwrap(), src);
    }

    #[test]
    fn zero_strength_is_noop() {
        let mut f = Field::new(16, 16).unwrap();
        f.add_radial_scale(8.0, 8.0, 6.0, 0.0, None).unwrap();
        f.add_pull(&PullSpec { x: 8.0, y: 8.0, vx: 1.0, vy: 0.0, amount: 0.0, radius: 6.0 }, None).unwrap();
        f.add_axis_squeeze(&SqueezeSpec { cx: 8.0, cy: 8.0, rx: 5.0, ry: 8.0, strength: 0.0, horizontal: true }, None).unwrap();
        f.add_segment_stretch(&StretchSpec { ax: 8.0, ay: 2.0, bx: 8.0, by: 14.0, width: 5.0, extend: 0.0 }, None).unwrap();
        assert_eq!(f.max_abs(), 0.0);
    }

    #[test]
    fn enlarge_samples_toward_centre() {
        // Bright dot right of centre must spread outward (content expands).
        let mut f = Field::new(32, 32).unwrap();
        f.add_radial_scale(16.0, 16.0, 12.0, 1.0, None).unwrap();
        // Sampling offset at (20,16) points back toward the centre (negative x).
        assert!(f.dx[16 * 32 + 20] < -0.05);
        // Pixel centres sit at half offsets: dy/dx must equal oy/ox exactly.
        let ratio = f.dy[16 * 32 + 20] / f.dx[16 * 32 + 20];
        assert!((ratio - 0.5 / 4.5).abs() < 1e-4, "ratio={ratio}");
        // Far corner untouched (compact support).
        assert_eq!(f.dx[0], 0.0);
        assert_eq!(f.dy[31 * 32 + 31], 0.0);
        // Remap level: dot energy moves outward.
        let mut src = vec![0.0f32; 32 * 32];
        src[16 * 32 + 20] = 1.0;
        let out = f.remap(&src, 1).unwrap();
        assert!(out[16 * 32 + 21] > 0.02, "outer neighbour lit");
        assert!(out[16 * 32 + 20] < 1.0, "centre dimmed");
    }

    #[test]
    fn pull_moves_content_along_direction() {
        // Content motion +x (jaw pulled inward): sampling offset must be -x.
        let mut f = Field::new(32, 32).unwrap();
        f.add_pull(&PullSpec { x: 16.0, y: 16.0, vx: 1.0, vy: 0.0, amount: 4.0, radius: 10.0 }, None).unwrap();
        let d = f.dx[16 * 32 + 16];
        assert!(d < -3.0 && d > -4.5, "dx={d}");
        assert!(f.dy[16 * 32 + 16].abs() < 1e-6);
    }

    #[test]
    fn squeeze_samples_outward_from_axis() {
        // Horizontal narrow: right of axis samples further right (positive dx).
        let mut f = Field::new(32, 32).unwrap();
        f.add_axis_squeeze(&SqueezeSpec { cx: 16.0, cy: 16.0, rx: 8.0, ry: 10.0, strength: 1.0, horizontal: true }, None).unwrap();
        assert!(f.dx[16 * 32 + 20] > 0.01);
        assert!(f.dx[16 * 32 + 12] < -0.01);
        assert!(f.dy[16 * 32 + 20].abs() < 1e-6);
    }

    #[test]
    fn stretch_moves_ankle_downward() {
        // Vertical leg axis hip(16,4) -> ankle(16,20), extend 20%: content below
        // the ankle moves down (+y), so sampling offsets point up (-y).
        let mut f = Field::new(32, 32).unwrap();
        f.add_segment_stretch(&StretchSpec { ax: 16.0, ay: 4.0, bx: 16.0, by: 20.0, width: 6.0, extend: 0.2 }, None).unwrap();
        // Just below the ankle: near-full downward content motion (dy ~= -3).
        let below_ankle = f.dy[21 * 32 + 16];
        assert!(below_ankle < -2.0, "dy={below_ankle}");
        // Above the hip: no motion.
        assert_eq!(f.dy[32 + 16], 0.0);
        // Far from the leg laterally: no motion.
        assert_eq!(f.dy[24 * 32 + 2], 0.0);
    }

    #[test]
    fn mask_gates_warp_for_static_background() {
        let mut full = vec![0.0f32; 16 * 16];
        for y in 0..16 {
            for x in 8..16 {
                full[y * 16 + x] = 1.0;
            }
        }
        let mut f = Field::new(16, 16).unwrap();
        f.add_pull(&PullSpec { x: 8.0, y: 8.0, vx: 1.0, vy: 0.0, amount: 5.0, radius: 12.0 }, Some(&full)).unwrap();
        // Masked-out (left) half stays exactly zero.
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(f.dx[y * 16 + x], 0.0);
                assert_eq!(f.dy[y * 16 + x], 0.0);
            }
        }
        // Masked-in half moves.
        assert!(f.dx[8 * 16 + 10].abs() > 0.5);
        // Remap level: background pixels bit-identical.
        let src: Vec<f32> = (0..16 * 16).map(|i| (i % 7) as f32 / 7.0).collect();
        let out = f.remap(&src, 1).unwrap();
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(out[y * 16 + x], src[y * 16 + x]);
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(Field::new(0, 8).is_err());
        assert!(Field::new(1_000_000, 1_000_000).is_err());
        let mut f = Field::new(8, 8).unwrap();
        assert!(f.add_pull(&PullSpec { x: 1.0, y: 1.0, vx: 1.0, vy: 0.0, amount: f32::NAN, radius: 4.0 }, None).is_err());
        assert!(f.add_pull(&PullSpec { x: 1.0, y: 1.0, vx: 0.0, vy: 0.0, amount: 1.0, radius: 4.0 }, None).is_ok()); // degenerate dir: no-op
        assert!(f.remap(&[0.0; 10], 3).is_err());
        assert!(f.remap(&[0.0; 8 * 8 * 3], 0).is_err());
        assert!(f.add_radial_scale(1.0, 1.0, 4.0, 0.5, Some(&[0.5; 3])).is_err());
    }
}
