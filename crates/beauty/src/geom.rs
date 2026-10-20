//! Landmark-driven geometry: faces/bodies become warp primitives + masks.
//!
//! All coordinates are normalized (0..1); builders scale to pixels. Pure
//! functions, no I/O, no inference: detectors (see `detect`) produce
//! [`FaceGeom`]/[`BodyGeom`], engines turn them into [`Primitive`]s and masks.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::warp::{Field, PullSpec, SqueezeSpec, StretchSpec};

/// Normalized 2D point (0..1 in image size).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn is_valid(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn dist(self, o: Point) -> f32 {
        (self.x - o.x).hypot(self.y - o.y)
    }
}

/// One detected face: dense landmarks when available, estimates otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceGeom {
    /// Normalized box [x0, y0, x1, y1].
    pub bbox: [f32; 4],
    pub left_eye: Point,
    pub right_eye: Point,
    /// Eye radius in units of image min-dimension.
    pub eye_radius: f32,
    pub nose_tip: Point,
    pub mouth_left: Point,
    pub mouth_right: Point,
    pub mouth_center: Point,
    /// Jawline polyline, ear-to-ear through the chin (may be sparse).
    pub jaw: Vec<Point>,
    pub chin: Point,
    /// Face oval polygon for skin masks (may be sparse).
    pub oval: Vec<Point>,
    pub has_dense: bool,
}

impl FaceGeom {
    /// Builds geometry from a box + 5 sparse points (YuNet-style) by
    /// estimating the jaw oval with frontal-face proportions.
    pub fn from_sparse(bbox: [f32; 4], left_eye: Point, right_eye: Point, nose_tip: Point, mouth_left: Point, mouth_right: Point) -> Result<Self, Error> {
        for (name, p) in [("left_eye", left_eye), ("right_eye", right_eye), ("nose_tip", nose_tip), ("mouth_left", mouth_left), ("mouth_right", mouth_right)] {
            if !p.is_valid() {
                return Err(Error::InvalidParam(format!("{name} not finite")));
            }
        }
        if bbox.iter().any(|v| !v.is_finite()) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            return Err(Error::InvalidParam("invalid face bbox".into()));
        }
        let (x0, y0, x1, y1) = (bbox[0], bbox[1], bbox[2], bbox[3]);
        let w = x1 - x0;
        let h = y1 - y0;
        let cx = (x0 + x1) * 0.5;
        let eye_dist = left_eye.dist(right_eye).max(1e-4);
        // Oval: inset ellipse polygon.
        let mut oval = Vec::with_capacity(40);
        for k in 0..40 {
            let t = 2.0 * core::f32::consts::PI * k as f32 / 40.0;
            oval.push(Point { x: cx + t.cos() * w * 0.44, y: (y0 + h * 0.52) + t.sin() * h * 0.44 });
        }
        // Jaw: lower arc of the oval, chin at the bottom.
        let jaw: Vec<Point> = oval.iter().copied().filter(|p| p.y > y0 + h * 0.55).collect();
        let chin = Point { x: cx, y: y1.min(y0 + h * 0.96) };
        let mouth_center = Point { x: (mouth_left.x + mouth_right.x) * 0.5, y: (mouth_left.y + mouth_right.y) * 0.5 };
        Ok(Self { bbox, left_eye, right_eye, eye_radius: eye_dist * 0.22, nose_tip, mouth_left, mouth_right, mouth_center, jaw, chin, oval, has_dense: false })
    }

    fn check(&self) -> Result<(), Error> {
        if self.bbox.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidParam("face bbox not finite".into()));
        }
        for (name, p) in
            [("left_eye", self.left_eye), ("right_eye", self.right_eye), ("nose_tip", self.nose_tip), ("mouth_center", self.mouth_center), ("chin", self.chin)]
        {
            if !p.is_valid() {
                return Err(Error::InvalidParam(format!("{name} not finite")));
            }
        }
        Ok(())
    }
}

/// Named body joints (COCO-17 subset actually used). Missing joints are `None`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Joints {
    pub nose: Option<Point>,
    pub left_shoulder: Option<Point>,
    pub right_shoulder: Option<Point>,
    pub left_hip: Option<Point>,
    pub right_hip: Option<Point>,
    pub left_knee: Option<Point>,
    pub right_knee: Option<Point>,
    pub left_ankle: Option<Point>,
    pub right_ankle: Option<Point>,
}

/// One detected body: joints + box. Person mask arrives separately (raster).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyGeom {
    /// Normalized box [x0, y0, x1, y1].
    pub bbox: [f32; 4],
    pub joints: Joints,
}

/// Face warp strengths, 0..100 each.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceWarpParams {
    pub eye_enlarge: f32,
    pub jaw_slim: f32,
    pub nose_refine: f32,
    /// N1 melon-seed / V-line shaping.
    pub vline: f32,
    /// N3 mouth-corner lift.
    pub mouth_lift: f32,
    /// N5 apple-muscle lift.
    pub cheek_lift: f32,
}

impl Default for FaceWarpParams {
    fn default() -> Self {
        Self { eye_enlarge: 0.0, jaw_slim: 0.0, nose_refine: 0.0, vline: 0.0, mouth_lift: 0.0, cheek_lift: 0.0 }
    }
}

/// Body warp strengths, 0..100 each.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyWarpParams {
    /// B1 torso slimming.
    pub slim_body: f32,
    /// B4-side waist narrowing (part of B1 region, own slider).
    pub waist_slim: f32,
    /// B2 leg slimming.
    pub slim_legs: f32,
    /// B3 leg lengthening (fraction of leg length, capped).
    pub leg_lengthen: f32,
    /// B5 head/body proportion (subtle head shrink).
    pub head_ratio: f32,
    /// B6 bust enlargement (teardrop).
    pub bust_size: f32,
    /// B6 bust lift.
    pub bust_lift: f32,
    /// B6 cleavage groove.
    pub cleavage: f32,
    /// B7 hip curve (hourglass).
    pub hip_curve: f32,
}

impl Default for BodyWarpParams {
    fn default() -> Self {
        Self {
            slim_body: 0.0,
            waist_slim: 0.0,
            slim_legs: 0.0,
            leg_lengthen: 0.0,
            head_ratio: 0.0,
            bust_size: 0.0,
            bust_lift: 0.0,
            cleavage: 0.0,
            hip_curve: 0.0,
        }
    }
}

/// A pixel-space warp primitive ready to apply to a [`Field`].
#[derive(Clone, Debug, PartialEq)]
pub enum Primitive {
    Radial { cx: f32, cy: f32, radius: f32, strength: f32 },
    Pull(PullSpec),
    Squeeze(SqueezeSpec),
    Stretch(StretchSpec),
}

impl Primitive {
    pub fn apply(&self, field: &mut Field, mask: Option<&[f32]>) -> Result<(), Error> {
        match self {
            Primitive::Radial { cx, cy, radius, strength } => field.add_radial_scale(*cx, *cy, *radius, *strength, mask),
            Primitive::Pull(spec) => field.add_pull(spec, mask),
            Primitive::Squeeze(spec) => field.add_axis_squeeze(spec, mask),
            Primitive::Stretch(spec) => field.add_segment_stretch(spec, mask),
        }
    }
}

fn s01(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 100.0) / 100.0 } else { 0.0 }
}

/// Builds face warp primitives (F4-F6 + N1-N4) in pixel space.
pub fn face_warps(face: &FaceGeom, img_w: usize, img_h: usize, params: &FaceWarpParams) -> Result<Vec<Primitive>, Error> {
    face.check()?;
    if img_w == 0 || img_h == 0 {
        return Err(Error::EmptyInput("face_warps image size"));
    }
    let (w, h) = (img_w as f32, img_h as f32);
    let px = |p: Point| (p.x * w, p.y * h);
    let min_dim = w.min(h);
    let face_w = (face.bbox[2] - face.bbox[0]).max(1e-4) * w;
    let cx = (face.bbox[0] + face.bbox[2]) * 0.5 * w;
    let mut out = Vec::new();

    // F4 eyes: magnify with a wide, soft falloff (a tight brush balloons the iris).
    let eye = s01(params.eye_enlarge);
    if eye > 0.0 {
        let r = (face.eye_radius * min_dim * 3.2).max(10.0);
        for e in [face.left_eye, face.right_eye] {
            let (ex, ey) = px(e);
            out.push(Primitive::Radial { cx: ex, cy: ey, radius: r, strength: eye * 0.55 });
        }
    }

    // F5/N1 lower-face narrowing, Liquify style: push the jawline along its own inward normal,
    // peaking at the gonial angle and fading to zero at the chin and at the ear. Pulling every
    // landmark toward the centre axis (what this used to do) flattens the contour into a cone;
    // a normal-direction push narrows the face while keeping the jaw's own curvature.
    let narrow = (s01(params.jaw_slim) * 0.6 + s01(params.vline) * 0.4).min(1.0);
    if narrow > 0.0 && face.jaw.len() >= 3 {
        let mut pts: Vec<Point> = face.jaw.iter().copied().filter(|p| p.is_valid()).collect();
        pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(core::cmp::Ordering::Equal));
        let top_y = pts.iter().fold(f32::INFINITY, |m, p| m.min(p.y)) * h;
        let chin_y = face.chin.y * h;
        let span = (chin_y - top_y).max(1.0);
        for (i, p) in pts.iter().enumerate() {
            let (jx, jy) = px(*p);
            let t = ((jy - top_y) / span).clamp(0.0, 1.0);
            let weight = bump(t, 0.45, 0.45);
            if weight <= 0.0 {
                continue;
            }
            let (tx, ty) = jaw_tangent(&pts, i, w, h);
            let len = tx.hypot(ty).max(1e-6);
            let (mut nx, mut ny) = (-ty / len, tx / len);
            if nx * (cx - jx) < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            out.push(Primitive::Pull(PullSpec { x: jx, y: jy, vx: nx, vy: ny * 0.75, amount: face_w * 0.040 * narrow * weight, radius: face_w * 0.42 }));
        }
    }

    // F6/N2 nose: gentle horizontal narrowing.
    let nose = s01(params.nose_refine);
    if nose > 0.0 {
        let eye_dist = face.left_eye.dist(face.right_eye).max(1e-4) * min_dim;
        let (nx, ny) = px(face.nose_tip);
        out.push(Primitive::Squeeze(SqueezeSpec {
            cx: nx,
            cy: ny,
            rx: eye_dist * 0.45,
            ry: eye_dist * 0.85,
            strength: nose * 0.85,
            horizontal: true,
            angle_rad: 0.0,
        }));
    }

    // N3 mouth corners: a small up-and-out lift.
    let lift = s01(params.mouth_lift);
    if lift > 0.0 {
        let mw = face.mouth_left.dist(face.mouth_right).max(1e-4) * w;
        for (corner, side) in [(face.mouth_left, -1.0), (face.mouth_right, 1.0)] {
            let (mx, my) = px(corner);
            out.push(Primitive::Pull(PullSpec { x: mx, y: my, vx: side * 0.5, vy: -0.9, amount: mw * 0.022 * lift, radius: mw * 0.6 }));
        }
    }

    // N5 apple cheeks: a soft upward lift at both cheek centres.
    let cheeks = s01(params.cheek_lift);
    if cheeks > 0.0 {
        let eye_dist = face.left_eye.dist(face.right_eye).max(1e-4);
        for (eye, side) in [(face.left_eye, -1.0), (face.right_eye, 1.0)] {
            let c = Point { x: eye.x, y: face.nose_tip.y + eye_dist * 0.10 };
            let (ccx, ccy) = px(c);
            out.push(Primitive::Pull(PullSpec {
                x: ccx,
                y: ccy,
                vx: side * 0.3,
                vy: -1.0,
                amount: eye_dist * min_dim * 0.022 * cheeks,
                radius: eye_dist * min_dim * 0.95,
            }));
        }
    }
    Ok(out)
}

/// Smooth bump: 1 at `center`, 0 at `center ± width`, C1-continuous.
fn bump(t: f32, center: f32, width: f32) -> f32 {
    if !t.is_finite() || width <= 0.0 {
        return 0.0;
    }
    let d = (t - center).abs();
    if d >= width {
        return 0.0;
    }
    0.5 + 0.5 * (core::f32::consts::PI * d / width).cos()
}

/// Tangent of a polyline at `i` (central difference over the neighbours), in pixels.
fn jaw_tangent(pts: &[Point], i: usize, w: f32, h: f32) -> (f32, f32) {
    let a = pts[i.saturating_sub(1)];
    let b = pts[(i + 1).min(pts.len() - 1)];
    let tx = (b.x - a.x) * w;
    let ty = (b.y - a.y) * h;
    if tx.hypot(ty) <= 1e-6 {
        return (1.0, 0.0);
    }
    (tx, ty)
}

/// Builds body warp primitives (B1-B3, B5) in pixel space.
/// Leg/torso work should be gated with the person mask by the caller.
pub fn body_warps(body: &BodyGeom, img_w: usize, img_h: usize, params: &BodyWarpParams) -> Result<Vec<Primitive>, Error> {
    if body.bbox.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidParam("body bbox not finite".into()));
    }
    if img_w == 0 || img_h == 0 {
        return Err(Error::EmptyInput("body_warps image size"));
    }
    let (w, h) = (img_w as f32, img_h as f32);
    let px = |p: Point| (p.x * w, p.y * h);
    let mut out = Vec::new();
    let j = &body.joints;
    // Torso axis + width from shoulders/hips.
    if let (Some(ls), Some(rs), Some(lh), Some(rh)) = (j.left_shoulder, j.right_shoulder, j.left_hip, j.right_hip)
        && ls.is_valid()
        && rs.is_valid()
        && lh.is_valid()
        && rh.is_valid()
    {
        let sh_c = Point { x: (ls.x + rs.x) * 0.5, y: (ls.y + rs.y) * 0.5 };
        let hip_c = Point { x: (lh.x + rh.x) * 0.5, y: (lh.y + rh.y) * 0.5 };
        let torso_w = ls.dist(rs).max(lh.dist(rh)).max(1e-4) * w;
        let torso_h = ((sh_c.y - hip_c.y).abs().max(1e-4)) * h;
        // B1 torso slim: horizontal squeeze on the torso axis.
        let slim = s01(params.slim_body);
        if slim > 0.0 {
            let (ccx, ccy) = px(Point { x: (sh_c.x + hip_c.x) * 0.5, y: (sh_c.y + hip_c.y) * 0.5 });
            out.push(Primitive::Squeeze(SqueezeSpec {
                cx: ccx,
                cy: ccy,
                rx: torso_w * 0.62,
                ry: torso_h * 0.62,
                strength: slim,
                horizontal: true,
                angle_rad: 0.0,
            }));
        }
        // Waist: narrower band between chest and hips.
        let waist = s01(params.waist_slim);
        if waist > 0.0 {
            let waist_c = Point { x: sh_c.x * 0.35 + hip_c.x * 0.65, y: sh_c.y * 0.35 + hip_c.y * 0.65 };
            let (ccx, ccy) = px(waist_c);
            out.push(Primitive::Squeeze(SqueezeSpec {
                cx: ccx,
                cy: ccy,
                rx: torso_w * 0.55,
                ry: torso_h * 0.20,
                strength: waist,
                horizontal: true,
                angle_rad: 0.0,
            }));
        }
        // B6 bust: a teardrop, not a hemisphere — an upper, rounder bulge plus a stronger lower one
        // that pushes the lower pole down and out, an optional lift, and a centre groove.
        let bust = s01(params.bust_size);
        let lift = s01(params.bust_lift);
        let cleave = s01(params.cleavage);
        let sh_w = ls.dist(rs).max(1e-4) * w;
        let sh_y = sh_c.y * h;
        let hip_y = hip_c.y * h;
        let axis_x = (sh_c.x + hip_c.x) * 0.5 * w;
        let bust_y = sh_y + (hip_y - sh_y) * 0.30;
        if bust > 0.0 || lift > 0.0 || cleave > 0.0 {
            let off = sh_w * 0.20;
            let r = sh_w * 0.26;
            for side in [-1.0f32, 1.0] {
                let bx = axis_x + side * off;
                if bust > 0.0 {
                    out.push(Primitive::Radial { cx: bx, cy: bust_y - r * 0.30, radius: r * 0.95, strength: bust * 0.16 });
                    out.push(Primitive::Radial { cx: bx, cy: bust_y + r * 0.35, radius: r * 0.85, strength: bust * 0.24 });
                }
                if lift > 0.0 {
                    out.push(Primitive::Pull(PullSpec { x: bx, y: bust_y + r * 0.25, vx: 0.0, vy: -1.0, amount: r * 0.30 * lift, radius: r * 1.15 }));
                }
                if cleave > 0.0 {
                    out.push(Primitive::Pull(PullSpec { x: bx - side * r * 0.55, y: bust_y, vx: -side, vy: 0.0, amount: r * 0.16 * cleave, radius: r * 0.55 }));
                }
            }
        }
        // B7 hips: a gentle outward curve for an hourglass silhouette against the waist.
        let hipc = s01(params.hip_curve);
        if hipc > 0.0 {
            let r = sh_w * 0.30;
            for (hip, side) in [(lh, -1.0f32), (rh, 1.0)] {
                let (hx, hy) = px(hip);
                out.push(Primitive::Radial { cx: hx + side * r * 0.35, cy: hy, radius: r, strength: hipc * 0.14 });
            }
        }
    }
    // Legs: slim (rotated squeezes) + lengthen (segment stretches).
    let legs = [(j.left_hip, j.left_knee, j.left_ankle), (j.right_hip, j.right_knee, j.right_ankle)];
    let slim_legs = s01(params.slim_legs);
    let lengthen = s01(params.leg_lengthen);
    for (hip, knee, ankle) in legs {
        let (Some(hip), Some(knee), Some(ankle)) = (hip, knee, ankle) else {
            continue;
        };
        if !(hip.is_valid() && knee.is_valid() && ankle.is_valid()) {
            continue;
        }
        if slim_legs > 0.0 {
            // Thigh + calf squeezes perpendicular to each segment.
            for (a, b) in [(hip, knee), (knee, ankle)] {
                let seg = a.dist(b).max(1e-4);
                let ang = (b.y - a.y).atan2(b.x - a.x);
                let mid = Point { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
                let (ccx, ccy) = px(mid);
                // Squeeze axis runs across the limb: rotate by segment angle.
                out.push(Primitive::Squeeze(SqueezeSpec {
                    cx: ccx,
                    cy: ccy,
                    rx: seg * w.min(h) * 0.55,
                    ry: seg * w.min(h) * 0.30,
                    strength: slim_legs,
                    horizontal: true,
                    angle_rad: ang,
                }));
            }
        }
        if lengthen > 0.0 {
            // Split extension proportional to segment lengths.
            let thigh = hip.dist(knee).max(1e-4);
            let calf = knee.dist(ankle).max(1e-4);
            let total = (thigh + calf).max(1e-4);
            let width = thigh * w.min(h) * 0.55;
            for (a, b, part) in [(hip, knee, thigh / total), (knee, ankle, calf / total)] {
                let (ax, ay) = px(a);
                let (bx, by) = px(b);
                out.push(Primitive::Stretch(StretchSpec { ax, ay, bx, by, width, extend: lengthen * 0.5 * part * 2.0 }));
            }
        }
    }
    // B5 head/body proportion: subtle shrink about the head centre.
    let head = s01(params.head_ratio);
    if head > 0.0
        && let Some(nose) = j.nose
        && nose.is_valid()
    {
        let (nx, ny) = px(nose);
        let r = (body.bbox[2] - body.bbox[0]).max(0.02) * w * 0.9;
        out.push(Primitive::Radial { cx: nx, cy: ny - r * 0.25, radius: r, strength: -0.5 * head });
    }
    Ok(out)
}

/// Skin regions for mask building: face oval + organ cutout polygons,
/// all normalized. Sparse faces get estimated rectangles-as-polygons.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinRegions {
    pub oval: Vec<Point>,
    pub cutouts: Vec<Vec<Point>>,
}
/// N5 cheek polygons + nasolabial fold segments.
pub type CheekRegions = (Vec<Vec<Point>>, Vec<(Point, Point)>);

impl FaceGeom {
    pub fn skin_regions(&self) -> Result<SkinRegions, Error> {
        self.check()?;
        let rect = |cx: f32, cy: f32, rw: f32, rh: f32| {
            vec![Point { x: cx - rw, y: cy - rh }, Point { x: cx + rw, y: cy - rh }, Point { x: cx + rw, y: cy + rh }, Point { x: cx - rw, y: cy + rh }]
        };
        let eye_w = self.left_eye.dist(self.right_eye).max(1e-4);
        let cutouts = vec![
            rect(self.left_eye.x, self.left_eye.y, eye_w * 0.22, eye_w * 0.16),
            rect(self.right_eye.x, self.right_eye.y, eye_w * 0.22, eye_w * 0.16),
            rect(self.left_eye.x, self.left_eye.y - eye_w * 0.28, eye_w * 0.26, eye_w * 0.12),
            rect(self.right_eye.x, self.right_eye.y - eye_w * 0.28, eye_w * 0.26, eye_w * 0.12),
            rect(self.mouth_center.x, self.mouth_center.y, eye_w * 0.34, eye_w * 0.17),
        ];
        Ok(SkinRegions { oval: self.oval.clone(), cutouts })
    }

    /// N5 cheek polygons + nasolabial fold segments.
    pub fn cheek_regions(&self) -> Result<CheekRegions, Error> {
        self.check()?;
        let eye_w = self.left_eye.dist(self.right_eye).max(1e-4);
        let mut cheeks = Vec::new();
        for ex in [self.left_eye.x, self.right_eye.x] {
            let (ccx, ccy) = (ex, self.nose_tip.y + eye_w * 0.10);
            let (rx, ry) = (eye_w * 0.30, eye_w * 0.34);
            let mut poly = Vec::with_capacity(8);
            for k in 0..8 {
                let t = 2.0 * core::f32::consts::PI * k as f32 / 8.0;
                poly.push(Point { x: ccx + t.cos() * rx, y: ccy + t.sin() * ry });
            }
            cheeks.push(poly);
        }
        // Fold: nose wing toward mouth corner on each side.
        let wing_y = self.nose_tip.y + eye_w * 0.06;
        let folds = vec![
            (Point { x: self.nose_tip.x - eye_w * 0.10, y: wing_y }, Point { x: self.mouth_left.x + eye_w * 0.06, y: self.mouth_left.y - eye_w * 0.04 }),
            (Point { x: self.nose_tip.x + eye_w * 0.10, y: wing_y }, Point { x: self.mouth_right.x - eye_w * 0.06, y: self.mouth_right.y - eye_w * 0.04 }),
        ];
        Ok((cheeks, folds))
    }
}

/// Capsule (stadium) mask around segment AB with `radius` (pixels) and soft
/// edge `feather` (pixels): 1 inside, smooth falloff, 0 outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StripSpec {
    pub ax: f32,
    pub ay: f32,
    pub bx: f32,
    pub by: f32,
    pub radius: f32,
    pub feather: f32,
}

pub fn segment_strip_mask(w: usize, h: usize, spec: &StripSpec) -> Result<Vec<f32>, Error> {
    let StripSpec { ax, ay, bx, by, radius, feather } = *spec;
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("segment_strip_mask dimensions"));
    }
    if ![ax, ay, bx, by, radius, feather].iter().all(|v| v.is_finite()) {
        return Err(Error::InvalidParam("strip args must be finite".into()));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam("mask dimensions overflow".to_string()))?;
    let mut out = vec![0.0f32; pixels];
    if radius <= 0.0 {
        return Ok(out);
    }
    let abx = bx - ax;
    let aby = by - ay;
    let len2 = (abx * abx + aby * aby).max(1e-8);
    let feather = feather.max(1.0);
    for y in 0..h {
        for x in 0..w {
            let px = x as f32 + 0.5 - ax;
            let py = y as f32 + 0.5 - ay;
            let t = ((px * abx + py * aby) / len2).clamp(0.0, 1.0);
            let dx = px - t * abx;
            let dy = py - t * aby;
            let d = dx.hypot(dy);
            out[y * w + x] = ((radius + feather - d) / feather).clamp(0.0, 1.0);
        }
    }
    Ok(out)
}

/// Classic Chai&Ngan YCrCb skin rule on interleaved RGB `f32` (0 otherwise).
/// Returns a hard 0/1 mask; callers feather it. Alpha ignored.
pub fn skin_chroma_mask(img: &[f32], w: usize, h: usize, ch: usize) -> Result<Vec<f32>, Error> {
    if w == 0 || h == 0 || ch == 0 {
        return Err(Error::EmptyInput("skin_chroma_mask input"));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam("image dimensions overflow".to_string()))?;
    if img.len() != pixels * ch {
        return Err(Error::ShapeMismatch { what: "skin_chroma_mask input", expected: pixels * ch, found: img.len() });
    }
    if ch < 3 {
        return Ok(vec![0.0; pixels]);
    }
    let mut out = vec![0.0f32; pixels];
    for i in 0..pixels {
        let (r, g, b) = (img[i * ch] * 255.0, img[i * ch + 1] * 255.0, img[i * ch + 2] * 255.0);
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        if (77.0..127.0).contains(&cb) && (133.0..173.0).contains(&cr) {
            out[i] = 1.0;
        }
    }
    Ok(out)
}

/// Smooth ramp: 0 outside `[lo - margin, hi + margin]`, 1 inside `[lo, hi]`, C1 between.
fn ramp(v: f32, lo: f32, hi: f32, margin: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    if v >= lo && v <= hi {
        return 1.0;
    }
    if v <= lo - margin || v >= hi + margin {
        return 0.0;
    }
    let t = if v < lo { (v - (lo - margin)) / margin } else { ((hi + margin) - v) / margin };
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft skin-likeness in 0..1 per pixel: the Chai & Ngan YCbCr ranges with smooth edges, times a
/// minimum-chroma gate (so grey and near-grey pixels score 0). Alpha is ignored.
///
/// The hard 0/1 [`skin_chroma_mask`] leaves holes where skin is shadowed or slightly off; a soft
/// score keeps those pixels partially in, which is what a beauty mask wants.
pub fn skin_chroma_score(img: &[f32], w: usize, h: usize, ch: usize) -> Result<Vec<f32>, Error> {
    if w == 0 || h == 0 || ch == 0 {
        return Err(Error::EmptyInput("skin_chroma_score input"));
    }
    let pixels = w.checked_mul(h).ok_or_else(|| Error::InvalidParam("image dimensions overflow".to_string()))?;
    if img.len() != pixels * ch {
        return Err(Error::ShapeMismatch { what: "skin_chroma_score input", expected: pixels * ch, found: img.len() });
    }
    if ch < 3 {
        return Ok(vec![0.0; pixels]);
    }
    let mut out = vec![0.0f32; pixels];
    for (i, v) in out.iter_mut().enumerate() {
        let (r, g, b) = (img[i * ch] * 255.0, img[i * ch + 1] * 255.0, img[i * ch + 2] * 255.0);
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        let spread = r.max(g).max(b) - r.min(g).min(b);
        let chroma = ((spread - 6.0) / 10.0).clamp(0.0, 1.0);
        let chroma = chroma * chroma * (3.0 - 2.0 * chroma);
        *v = ramp(cb, 77.0, 127.0, 9.0) * ramp(cr, 133.0, 173.0, 9.0) * chroma;
    }
    Ok(out)
}

/// Estimated leg joints (hip/knee/ankle per leg) from a person mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegPair {
    pub left_hip: Point,
    pub left_knee: Point,
    pub left_ankle: Point,
    pub right_hip: Point,
    pub right_knee: Point,
    pub right_ankle: Point,
}

/// Estimates leg joints from a person probability mask (top-left origin).
///
/// Frontal-standing heuristic: binarizes at 0.5, finds the crotch row where
/// the silhouette splits into two columns, then places hip/knee/ankle rows.
/// Returns None when no split exists (side pose, cropped, empty).
pub fn estimate_legs_from_mask(mask: &[f32], w: usize, h: usize) -> Result<Option<LegPair>, Error> {
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("legs mask dimensions"));
    }
    if mask.len() != w * h {
        return Err(Error::ShapeMismatch { what: "legs mask input", expected: w * h, found: mask.len() });
    }
    let run_min = (w / 128).max(2);
    let mut runs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(h);
    for y in 0..h {
        let mut row_runs = Vec::new();
        let mut x = 0;
        while x < w {
            if mask[y * w + x] >= 0.5 {
                let start = x;
                while x < w && mask[y * w + x] >= 0.5 {
                    x += 1;
                }
                if x - start >= run_min {
                    row_runs.push((start, x));
                }
            } else {
                x += 1;
            }
        }
        runs.push(row_runs);
    }
    let mass_rows: Vec<usize> = (0..h).filter(|y| !runs[*y].is_empty()).collect();
    if mass_rows.is_empty() {
        return Ok(None);
    }
    let top = mass_rows[0];
    let bottom = mass_rows[mass_rows.len() - 1];
    let height = bottom.saturating_sub(top);
    if height < 20 {
        return Ok(None);
    }
    let search_from = top + height * 25 / 100;
    let search_to = bottom.saturating_sub(height / 10);
    let mut crotch = None;
    for (y, row) in runs.iter().enumerate().take(search_to.min(h)).skip(search_from) {
        if row.len() >= 2 {
            crotch = Some(y);
            break;
        }
    }
    let crotch = match crotch {
        Some(y) => y,
        None => return Ok(None),
    };
    let legs_h = bottom.saturating_sub(crotch);
    if legs_h * 100 < height * 15 {
        return Ok(None);
    }
    let mut sx = 0usize;
    let mut n = 0usize;
    for row_runs in runs.iter() {
        for (a, b) in row_runs {
            for x in *a..*b {
                sx += x;
                n += 1;
            }
        }
    }
    if n == 0 {
        return Ok(None);
    }
    let cx = sx as f32 / n as f32;
    let (mut lsx, mut ln, mut rsx, mut rn) = (0usize, 0usize, 0usize, 0usize);
    for row_runs in runs.iter().take(bottom.min(h - 1) + 1).skip(crotch) {
        for (a, b) in row_runs {
            for x in *a..*b {
                if (x as f32) < cx {
                    lsx += x;
                    ln += 1;
                } else {
                    rsx += x;
                    rn += 1;
                }
            }
        }
    }
    if ln == 0 || rn == 0 {
        return Ok(None);
    }
    let (left_cx, right_cx) = (lsx as f32 / ln as f32, rsx as f32 / rn as f32);
    let hip_y = crotch.saturating_sub(height * 4 / 100) as f32 + 0.5;
    let knee_y = crotch as f32 + legs_h as f32 * 0.42;
    let ankle_y = (bottom as f32 - height as f32 * 0.02).max(knee_y + 1.0);
    let nx = |x: f32| (x / w as f32).clamp(0.0, 1.0);
    let ny = |y: f32| (y / h as f32).clamp(0.0, 1.0);
    Ok(Some(LegPair {
        left_hip: Point { x: nx(left_cx), y: ny(hip_y) },
        left_knee: Point { x: nx(left_cx), y: ny(knee_y) },
        left_ankle: Point { x: nx(left_cx), y: ny(ankle_y) },
        right_hip: Point { x: nx(right_cx), y: ny(hip_y) },
        right_knee: Point { x: nx(right_cx), y: ny(knee_y) },
        right_ankle: Point { x: nx(right_cx), y: ny(ankle_y) },
    }))
}

/// Fills missing hip/knee/ankle joints from a person mask; detected joints win.
pub fn complete_leg_joints(joints: &Joints, mask: &[f32], w: usize, h: usize) -> Result<Joints, Error> {
    let mut out = joints.clone();
    let need = out.left_hip.is_none()
        || out.right_hip.is_none()
        || out.left_knee.is_none()
        || out.right_knee.is_none()
        || out.left_ankle.is_none()
        || out.right_ankle.is_none();
    if !need {
        return Ok(out);
    }
    if let Some(legs) = estimate_legs_from_mask(mask, w, h)? {
        out.left_hip = out.left_hip.or(Some(legs.left_hip));
        out.right_hip = out.right_hip.or(Some(legs.right_hip));
        out.left_knee = out.left_knee.or(Some(legs.left_knee));
        out.right_knee = out.right_knee.or(Some(legs.right_knee));
        out.left_ankle = out.left_ankle.or(Some(legs.left_ankle));
        out.right_ankle = out.right_ankle.or(Some(legs.right_ankle));
    }
    Ok(out)
}

/// Torso anchors estimated from a person silhouette (normalized coords).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsoEst {
    /// Silhouette centre x over the torso band.
    pub center_x: f32,
    pub shoulder_y: f32,
    /// Half-width of the silhouette at the shoulder row (fraction of w).
    pub shoulder_hw: f32,
    pub hip_y: f32,
    pub hip_hw: f32,
    /// Last silhouette row (normalized).
    pub bottom_y: f32,
}

fn silhouette_runs(mask: &[f32], w: usize, h: usize) -> Vec<Vec<(usize, usize)>> {
    let run_min = (w / 128).max(2);
    let mut runs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(h);
    for y in 0..h {
        let mut row_runs = Vec::new();
        let mut x = 0;
        while x < w {
            if mask[y * w + x] >= 0.5 {
                let start = x;
                while x < w && mask[y * w + x] >= 0.5 {
                    x += 1;
                }
                if x - start >= run_min {
                    row_runs.push((start, x));
                }
            } else {
                x += 1;
            }
        }
        runs.push(row_runs);
    }
    runs
}

fn widest_run(runs: &[Vec<(usize, usize)>], y: usize) -> Option<(usize, usize)> {
    runs.get(y).and_then(|row| row.iter().max_by_key(|(a, b)| b - a).copied())
}

/// Estimates torso anchors from a person probability mask given the neck
/// position (face chin, normalized). Returns None for headshots (no torso
/// below the neck) and degenerate masks.
pub fn estimate_torso_from_mask(mask: &[f32], w: usize, h: usize, neck: Point) -> Result<Option<TorsoEst>, Error> {
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("torso mask dimensions"));
    }
    if mask.len() != w * h {
        return Err(Error::ShapeMismatch { what: "torso mask input", expected: w * h, found: mask.len() });
    }
    if !neck.is_valid() {
        return Ok(None);
    }
    let runs = silhouette_runs(mask, w, h);
    let neck_row = ((neck.y * h as f32) as usize).min(h - 1);
    // Shoulder row: just below the neck; tolerate small gaps above the torso.
    let mut shoulder_row = None;
    for (y, row) in runs.iter().enumerate().take((neck_row + h * 8 / 100 + 1).min(h)).skip(neck_row) {
        if !row.is_empty() {
            shoulder_row = Some(y.max(neck_row + 1).min(h - 1));
            break;
        }
    }
    let shoulder_row = match shoulder_row {
        Some(y) => y,
        None => return Ok(None),
    };
    let mut bottom = None;
    for y in (shoulder_row..h).rev() {
        if !runs[y].is_empty() {
            bottom = Some(y);
            break;
        }
    }
    let bottom = match bottom {
        Some(y) => y,
        None => return Ok(None),
    };
    let torso_h = bottom.saturating_sub(shoulder_row);
    if torso_h < 8 || torso_h * 100 < h * 4 {
        return Ok(None);
    }
    // Centre x: mass centroid over the torso band.
    let (mut sx, mut n) = (0usize, 0usize);
    for row in runs.iter().take(bottom + 1).skip(shoulder_row) {
        for (a, b) in row {
            for x in *a..*b {
                sx += x;
                n += 1;
            }
        }
    }
    if n == 0 {
        return Ok(None);
    }
    let cx = sx as f32 / n as f32;
    let sho_w = widest_run(&runs, shoulder_row).map_or(0, |(a, b)| b - a) as f32;
    let hip_row = shoulder_row + torso_h * 28 / 100;
    let hip_w = widest_run(&runs, hip_row).map_or(0, |(a, b)| b - a) as f32;
    if sho_w <= 0.0 || hip_w <= 0.0 {
        return Ok(None);
    }
    Ok(Some(TorsoEst {
        center_x: (cx / w as f32).clamp(0.0, 1.0),
        shoulder_y: (shoulder_row as f32 / h as f32).clamp(0.0, 1.0),
        shoulder_hw: (sho_w * 0.5 / w as f32).clamp(0.0, 0.5),
        hip_y: (hip_row as f32 / h as f32).clamp(0.0, 1.0),
        hip_hw: (hip_w * 0.5 / w as f32).clamp(0.0, 0.5),
        bottom_y: (bottom as f32 / h as f32).clamp(0.0, 1.0),
    }))
}

/// Estimates leg joints from an *unsplit* lower silhouette (legs together,
/// long coat, side pose): both legs share the silhouette centre with narrow
/// offsets. Good enough for slimming; lengthening must stay disabled.
pub fn estimate_legs_unsplit_from_mask(mask: &[f32], w: usize, h: usize, torso: &TorsoEst) -> Result<Option<LegPair>, Error> {
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("unsplit legs mask dimensions"));
    }
    if mask.len() != w * h {
        return Err(Error::ShapeMismatch { what: "unsplit legs mask input", expected: w * h, found: mask.len() });
    }
    let hip_y = (torso.hip_y * h as f32) as usize;
    let bottom = (torso.bottom_y * h as f32) as usize;
    if bottom <= hip_y + 6 {
        return Ok(None);
    }
    let knee_y = hip_y + (bottom - hip_y) / 2;
    let ankle_y = bottom.saturating_sub(h / 100 + 1);
    if !(hip_y < knee_y && knee_y < ankle_y) {
        return Ok(None);
    }
    let off = (torso.hip_hw * 0.45).clamp(0.005, 0.2);
    let cx = torso.center_x.clamp(0.0, 1.0);
    let ny = |y: usize| (y as f32 / h as f32).clamp(0.0, 1.0);
    Ok(Some(LegPair {
        left_hip: Point { x: (cx - off).clamp(0.0, 1.0), y: ny(hip_y) },
        left_knee: Point { x: (cx - off).clamp(0.0, 1.0), y: ny(knee_y) },
        left_ankle: Point { x: (cx - off).clamp(0.0, 1.0), y: ny(ankle_y) },
        right_hip: Point { x: (cx + off).clamp(0.0, 1.0), y: ny(hip_y) },
        right_knee: Point { x: (cx + off).clamp(0.0, 1.0), y: ny(knee_y) },
        right_ankle: Point { x: (cx + off).clamp(0.0, 1.0), y: ny(ankle_y) },
    }))
}

fn shoulders_sane(j: &Joints, neck_y: f32) -> bool {
    match (j.left_shoulder, j.right_shoulder) {
        (Some(l), Some(r)) if l.is_valid() && r.is_valid() => {
            let dy = (l.y - r.y).abs();
            let width = (l.x - r.x).abs();
            l.y > neck_y - 0.02 && l.y < neck_y + 0.15 && r.y > neck_y - 0.02 && r.y < neck_y + 0.15 && dy < 0.08 && width > 0.02
        }
        _ => false,
    }
}

fn hips_sane(j: &Joints, neck_y: f32, shoulder_y: f32) -> bool {
    match (j.left_hip, j.right_hip) {
        (Some(l), Some(r)) if l.is_valid() && r.is_valid() => {
            l.y > neck_y + 0.03 && r.y > neck_y + 0.03 && l.y > shoulder_y && r.y > shoulder_y && (l.y - r.y).abs() < 0.10
        }
        _ => false,
    }
}

fn legs_detected(j: &Joints) -> bool {
    [j.left_hip, j.left_knee, j.left_ankle, j.right_hip, j.right_knee, j.right_ankle].into_iter().all(|p| p.is_some_and(|p| p.is_valid()))
}

/// Completes body joints for warping from detected joints + person mask.
/// Detected joints win when geometrically sane; otherwise mask-based torso
/// and leg estimates fill the gaps so body shaping works even when the pose
/// backend only returns upper-body joints.
///
/// Returns the completed joints plus whether the legs came from a *split*
/// silhouette (or full detections): leg *lengthening* is only safe then,
/// while slimming also works on the unsplit fallback.
pub fn complete_body_joints(joints: &Joints, mask: Option<&[f32]>, w: usize, h: usize, neck: Point) -> Result<(Joints, bool), Error> {
    let mut out = joints.clone();
    let mut legs_split = legs_detected(&out);
    let Some(mask) = mask else {
        return Ok((out, legs_split));
    };
    if let Some(legs) = estimate_legs_from_mask(mask, w, h)? {
        legs_split = true;
        out.left_hip = out.left_hip.or(Some(legs.left_hip));
        out.right_hip = out.right_hip.or(Some(legs.right_hip));
        out.left_knee = out.left_knee.or(Some(legs.left_knee));
        out.right_knee = out.right_knee.or(Some(legs.right_knee));
        out.left_ankle = out.left_ankle.or(Some(legs.left_ankle));
        out.right_ankle = out.right_ankle.or(Some(legs.right_ankle));
    }
    let torso = estimate_torso_from_mask(mask, w, h, neck)?;
    if let Some(t) = torso {
        if !shoulders_sane(&out, neck.y) {
            out.left_shoulder = Some(Point { x: (t.center_x - t.shoulder_hw * 0.9).clamp(0.0, 1.0), y: t.shoulder_y });
            out.right_shoulder = Some(Point { x: (t.center_x + t.shoulder_hw * 0.9).clamp(0.0, 1.0), y: t.shoulder_y });
        }
        let shoulder_y = out.left_shoulder.map_or(t.shoulder_y, |p| p.y);
        if !hips_sane(&out, neck.y, shoulder_y) {
            out.left_hip = Some(Point { x: (t.center_x - t.hip_hw * 0.9).clamp(0.0, 1.0), y: t.hip_y });
            out.right_hip = Some(Point { x: (t.center_x + t.hip_hw * 0.9).clamp(0.0, 1.0), y: t.hip_y });
        }
        if !legs_detected(&out)
            && let Some(fallback) = estimate_legs_unsplit_from_mask(mask, w, h, &t)?
        {
            out.left_hip = out.left_hip.or(Some(fallback.left_hip));
            out.right_hip = out.right_hip.or(Some(fallback.right_hip));
            out.left_knee = out.left_knee.or(Some(fallback.left_knee));
            out.right_knee = out.right_knee.or(Some(fallback.right_knee));
            out.left_ankle = out.left_ankle.or(Some(fallback.left_ankle));
            out.right_ankle = out.right_ankle.or(Some(fallback.right_ankle));
        }
    }
    Ok((out, legs_split))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_face() -> FaceGeom {
        FaceGeom::from_sparse(
            [0.3, 0.1, 0.7, 0.9],
            Point { x: 0.42, y: 0.38 },
            Point { x: 0.58, y: 0.38 },
            Point { x: 0.5, y: 0.55 },
            Point { x: 0.44, y: 0.70 },
            Point { x: 0.56, y: 0.70 },
        )
        .unwrap()
    }

    fn sample_body() -> BodyGeom {
        BodyGeom {
            bbox: [0.3, 0.05, 0.7, 0.98],
            joints: Joints {
                nose: Some(Point { x: 0.5, y: 0.10 }),
                left_shoulder: Some(Point { x: 0.42, y: 0.25 }),
                right_shoulder: Some(Point { x: 0.58, y: 0.25 }),
                left_hip: Some(Point { x: 0.44, y: 0.52 }),
                right_hip: Some(Point { x: 0.56, y: 0.52 }),
                left_knee: Some(Point { x: 0.44, y: 0.72 }),
                right_knee: Some(Point { x: 0.56, y: 0.72 }),
                left_ankle: Some(Point { x: 0.44, y: 0.94 }),
                right_ankle: Some(Point { x: 0.56, y: 0.94 }),
            },
        }
    }

    #[test]
    fn sparse_face_estimates_geometry() {
        let f = sample_face();
        assert!(!f.has_dense);
        assert!(!f.oval.is_empty());
        assert!(f.jaw.len() >= 4);
        assert!(f.eye_radius > 0.0);
        assert!(
            FaceGeom::from_sparse(
                [0.7, 0.1, 0.3, 0.9],
                Point { x: 0.4, y: 0.4 },
                Point { x: 0.6, y: 0.4 },
                Point { x: 0.5, y: 0.5 },
                Point { x: 0.4, y: 0.7 },
                Point { x: 0.6, y: 0.7 },
            )
            .is_err()
        );
    }

    #[test]
    fn face_warps_cover_all_sliders() {
        let f = sample_face();
        let p = FaceWarpParams { eye_enlarge: 50.0, jaw_slim: 50.0, nose_refine: 50.0, vline: 50.0, mouth_lift: 50.0, cheek_lift: 50.0 };
        let prims = face_warps(&f, 400, 500, &p).unwrap();
        let mut kinds = [0; 4];
        for prim in &prims {
            match prim {
                Primitive::Radial { .. } => kinds[0] += 1,
                Primitive::Pull(_) => kinds[1] += 1,
                Primitive::Squeeze(_) => kinds[2] += 1,
                Primitive::Stretch(_) => kinds[3] += 1,
            }
        }
        assert_eq!(kinds[0], 2, "two eye radials");
        assert!(kinds[1] >= 6, "jaw + vline + mouth + cheek pulls");
        assert_eq!(kinds[2], 1, "nose squeeze");
        assert_eq!(kinds[3], 0, "no stretches on faces");
        // Zero params => no primitives.
        assert!(face_warps(&f, 400, 500, &FaceWarpParams::default()).unwrap().is_empty());
        // V-line pulls point inward (toward centre x). Cheek/mouth lifts pull
        // upward (vy <= -0.5), so they are excluded from this check.
        for prim in prims {
            if let Primitive::Pull(spec) = prim
                && spec.amount > 0.0
                && spec.y > 250.0
                && spec.radius > 60.0
                && spec.vy > -0.5
            {
                let inward = if spec.x < 200.0 { spec.vx > 0.0 } else { spec.vx < 0.0 };
                assert!(inward, "v-line must pull inward: {spec:?}");
            }
        }
    }

    #[test]
    fn cheek_lift_pulls_upward() {
        let f = sample_face();
        let p = FaceWarpParams { cheek_lift: 100.0, ..FaceWarpParams::default() };
        let prims = face_warps(&f, 400, 500, &p).unwrap();
        assert_eq!(prims.len(), 2, "one pull per cheek");
        for prim in &prims {
            assert!(matches!(prim, Primitive::Pull(_)), "cheek lift must be pulls");
            if let Primitive::Pull(spec) = prim {
                assert!(spec.vy < 0.0, "must pull upward: {spec:?}");
                assert!(spec.amount > 0.0 && spec.radius > 0.0);
            }
        }
    }

    #[test]
    fn body_warps_cover_all_sliders() {
        let b = sample_body();
        let p = BodyWarpParams {
            slim_body: 50.0,
            waist_slim: 50.0,
            slim_legs: 50.0,
            leg_lengthen: 20.0,
            head_ratio: 50.0,
            bust_size: 40.0,
            bust_lift: 30.0,
            cleavage: 30.0,
            hip_curve: 30.0,
        };
        let prims = body_warps(&b, 400, 800, &p).unwrap();
        let mut squeeze = 0;
        let mut stretch = 0;
        let mut radial_out = 0;
        let mut radial_in = 0;
        let mut pull = 0;
        for prim in &prims {
            match prim {
                Primitive::Squeeze(_) => squeeze += 1,
                Primitive::Stretch(_) => stretch += 1,
                Primitive::Radial { strength, .. } => {
                    if *strength > 0.0 {
                        radial_out += 1;
                    } else {
                        radial_in += 1;
                    }
                }
                Primitive::Pull(_) => pull += 1,
            }
        }
        assert_eq!(squeeze, 2 + 2 * 2, "torso+waist+2 segs x2 legs");
        assert_eq!(stretch, 4, "thigh+calf x2 legs");
        assert_eq!(radial_in, 1, "head shrink is the only inward radial");
        assert_eq!(radial_out, 4 + 2, "two bust bulges per side, one radial per hip");
        assert_eq!(pull, 4, "bust lift + cleavage groove per side");
        assert!(body_warps(&b, 400, 800, &BodyWarpParams::default()).unwrap().is_empty());
        // Each new slider works on its own.
        for (p, want) in [
            (BodyWarpParams { bust_size: 50.0, ..BodyWarpParams::default() }, 4),
            (BodyWarpParams { bust_lift: 50.0, ..BodyWarpParams::default() }, 2),
            (BodyWarpParams { cleavage: 50.0, ..BodyWarpParams::default() }, 2),
            (BodyWarpParams { hip_curve: 50.0, ..BodyWarpParams::default() }, 2),
        ] {
            let got = body_warps(&b, 400, 800, &p).unwrap();
            assert_eq!(got.len(), want, "slider produced {got:?}");
        }
        // Missing joints degrade gracefully (no leg primitives, no error).
        let mut b2 = b.clone();
        b2.joints.left_knee = None;
        b2.joints.right_knee = None;
        let prims2 = body_warps(&b2, 400, 800, &p).unwrap();
        assert!(prims2.len() < prims.len());
    }

    #[test]
    fn skin_regions_and_cheeks_sane() {
        let f = sample_face();
        let regions = f.skin_regions().unwrap();
        assert!(regions.oval.len() >= 10);
        assert_eq!(regions.cutouts.len(), 5);
        let (cheeks, folds) = f.cheek_regions().unwrap();
        assert_eq!(cheeks.len(), 2);
        assert_eq!(folds.len(), 2);
        for cheek in &cheeks {
            assert_eq!(cheek.len(), 8);
            for p in cheek {
                assert!(p.is_valid());
            }
        }
    }

    #[test]
    fn strip_mask_capsule_shape() {
        let spec = StripSpec { ax: 4.0, ay: 10.0, bx: 16.0, by: 10.0, radius: 2.0, feather: 2.0 };
        let m = segment_strip_mask(20, 20, &spec).unwrap();
        assert_eq!(m[10 * 20 + 10], 1.0);
        assert_eq!(m[0], 0.0);
        // Soft edge between radius and radius+feather.
        let edge = m[10 * 20 + 4];
        assert!(edge > 0.0 && edge <= 1.0);
        let bad = StripSpec { ax: 0.0, ay: 0.0, bx: 1.0, by: 1.0, radius: 1.0, feather: 1.0 };
        assert!(segment_strip_mask(0, 5, &bad).is_err());
    }

    #[test]
    fn chroma_rule_detects_skin_not_sky() {
        // Skin-ish, sky-blue, pure green pixels.
        let img = vec![
            0.85, 0.65, 0.55, //
            0.35, 0.55, 0.85, //
            0.20, 0.70, 0.20, //
        ];
        let m = skin_chroma_mask(&img, 3, 1, 3).unwrap();
        assert_eq!(m, vec![1.0, 0.0, 0.0]);
        assert!(skin_chroma_mask(&[0.5; 4], 3, 1, 3).is_err());
    }

    #[test]
    fn primitives_apply_to_field() {
        let f = sample_face();
        let p = FaceWarpParams { eye_enlarge: 100.0, ..FaceWarpParams::default() };
        let prims = face_warps(&f, 64, 64, &p).unwrap();
        let mut field = Field::new(64, 64).unwrap();
        for prim in &prims {
            prim.apply(&mut field, None).unwrap();
        }
        assert!(field.max_abs() > 0.1);
    }

    fn person_mask_100() -> Vec<f32> {
        // Torso rect + two leg rects on 100x100.
        let mut m = vec![0.0f32; 100 * 100];
        for y in 10..52 {
            for x in 40..60 {
                m[y * 100 + x] = 1.0;
            }
        }
        for y in 52..96 {
            for x in 42..48 {
                m[y * 100 + x] = 1.0;
            }
            for x in 52..58 {
                m[y * 100 + x] = 1.0;
            }
        }
        m
    }

    #[test]
    fn legs_estimated_from_mask() {
        let m = person_mask_100();
        let legs = estimate_legs_from_mask(&m, 100, 100).unwrap().unwrap();
        assert!(legs.left_hip.x < 0.5 && legs.right_hip.x > 0.5);
        assert!((legs.left_hip.x - 0.45).abs() < 0.03);
        assert!((legs.right_hip.x - 0.55).abs() < 0.03);
        assert!(legs.left_hip.y < legs.left_knee.y && legs.left_knee.y < legs.left_ankle.y);
        assert!((legs.left_hip.y - 0.48).abs() < 0.06);
        assert!((legs.left_knee.y - 0.68).abs() < 0.07);
        assert!((legs.left_ankle.y - 0.93).abs() < 0.04);
    }

    #[test]
    fn legs_absent_without_split() {
        assert!(estimate_legs_from_mask(&vec![0.0; 100 * 100], 100, 100).unwrap().is_none());
        let mut single = vec![0.0f32; 100 * 100];
        for y in 10..90 {
            for x in 40..60 {
                single[y * 100 + x] = 1.0;
            }
        }
        assert!(estimate_legs_from_mask(&single, 100, 100).unwrap().is_none());
        assert!(estimate_legs_from_mask(&[0.5; 3], 100, 100).is_err());
    }

    #[test]
    fn complete_prefers_detected_joints() {
        let m = person_mask_100();
        let joints = Joints { left_hip: Some(Point { x: 0.1, y: 0.2 }), ..Joints::default() };
        let out = complete_leg_joints(&joints, &m, 100, 100).unwrap();
        assert_eq!(out.left_hip, Some(Point { x: 0.1, y: 0.2 }));
        assert!(out.right_hip.is_some() && out.left_knee.is_some() && out.right_ankle.is_some());
        assert!(out.right_hip.is_some() && out.left_knee.is_some() && out.right_ankle.is_some());
    }

    fn torso_mask_100() -> Vec<f32> {
        // Head blob + torso rect + single wide lower rect (legs together, NO split).
        let mut m = vec![0.0f32; 100 * 100];
        for y in 5..15 {
            for x in 42..58 {
                m[y * 100 + x] = 1.0;
            }
        }
        for y in 15..55 {
            for x in 38..62 {
                m[y * 100 + x] = 1.0;
            }
        }
        for y in 55..95 {
            for x in 40..60 {
                m[y * 100 + x] = 1.0;
            }
        }
        m
    }

    #[test]
    fn torso_est_sane_on_unsplit() {
        let m = torso_mask_100();
        let t = estimate_torso_from_mask(&m, 100, 100, Point { x: 0.5, y: 0.15 }).unwrap().unwrap();
        assert!((t.center_x - 0.5).abs() < 0.02, "cx={}", t.center_x);
        assert!((t.shoulder_y - 0.16).abs() < 0.03, "sho={}", t.shoulder_y);
        assert!(t.hip_y > t.shoulder_y && t.hip_y < 0.50, "hip={}", t.hip_y);
        assert!((t.shoulder_hw - 0.12).abs() < 0.03, "sho_hw={}", t.shoulder_hw);
        assert!(t.hip_hw > 0.05, "hip_hw={}", t.hip_hw);
        assert!((t.bottom_y - 0.94).abs() < 0.02, "bot={}", t.bottom_y);
        // Headshot: no torso below the neck.
        let mut head = vec![0.0f32; 100 * 100];
        for y in 5..20 {
            for x in 42..58 {
                head[y * 100 + x] = 1.0;
            }
        }
        assert!(estimate_torso_from_mask(&head, 100, 100, Point { x: 0.5, y: 0.15 }).unwrap().is_none());
        assert!(estimate_torso_from_mask(&[0.0; 4], 100, 100, Point { x: 0.5, y: 0.15 }).is_err());
    }

    #[test]
    fn unsplit_legs_ordered_inside_silhouette() {
        let m = torso_mask_100();
        let t = estimate_torso_from_mask(&m, 100, 100, Point { x: 0.5, y: 0.15 }).unwrap().unwrap();
        let legs = estimate_legs_unsplit_from_mask(&m, 100, 100, &t).unwrap().unwrap();
        assert!(legs.left_hip.y < legs.left_knee.y && legs.left_knee.y < legs.left_ankle.y);
        assert!(legs.left_hip.x < 0.5 && legs.right_hip.x > 0.5);
        assert!((legs.left_hip.x - 0.45).abs() < 0.06);
        // Stubby lower body: no legs.
        let short = TorsoEst { hip_y: 0.90, bottom_y: 0.92, ..t };
        assert!(estimate_legs_unsplit_from_mask(&m, 100, 100, &short).unwrap().is_none());
    }

    #[test]
    fn complete_body_replaces_insane_shoulders() {
        // Shoulders detected mid-torso (low-confidence junk): mask wins.
        let m = torso_mask_100();
        let joints = Joints {
            nose: Some(Point { x: 0.5, y: 0.10 }),
            left_shoulder: Some(Point { x: 0.40, y: 0.64 }),
            right_shoulder: Some(Point { x: 0.60, y: 0.64 }),
            ..Joints::default()
        };
        let (out, split) = complete_body_joints(&joints, Some(&m), 100, 100, Point { x: 0.5, y: 0.15 }).unwrap();
        assert!(!split, "no leg split in this mask");
        let sho = out.left_shoulder.unwrap();
        assert!(sho.y < 0.25, "shoulders re-estimated near neck: {sho:?}");
        assert!(out.left_hip.is_some() && out.right_knee.is_some() && out.left_ankle.is_some());
        assert_eq!(out.nose, joints.nose, "nose untouched");
    }

    #[test]
    fn complete_body_keeps_sane_split() {
        let b = sample_body();
        let m = person_mask_100();
        let (out, split) = complete_body_joints(&b.joints, Some(&m), 100, 100, Point { x: 0.5, y: 0.12 }).unwrap();
        assert!(split);
        assert_eq!(out, b.joints, "sane detections win everywhere");
    }

    #[test]
    fn complete_body_without_mask_keeps_input() {
        let b = sample_body();
        let (out, split) = complete_body_joints(&b.joints, None, 100, 100, Point { x: 0.5, y: 0.12 }).unwrap();
        assert_eq!(out, b.joints);
        assert!(split, "full detections allow lengthening");
        let empty = Joints::default();
        let (out2, split2) = complete_body_joints(&empty, None, 100, 100, Point { x: 0.5, y: 0.12 }).unwrap();
        assert_eq!(out2, empty);
        assert!(!split2);
    }
}
