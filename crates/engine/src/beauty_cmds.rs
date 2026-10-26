//! Portrait beautification: Filter › Beautify (Portrait Beautify / Skin / Reshape).
//!
//! Native engine front-end for the [`photocraft_beauty`] crate (Stage 6):
//! face/body detection runs once per photo (content-hashed cache), then the
//! cached landmarks drive skin, light and warp work. Slider ticks on the same
//! photo re-render from the cached detection instead of re-detecting.
//!
//! Why this module lives in `engine` instead of `algo::FilterParams`: `algo`
//! and `beauty` are both L2, and sideways (same-layer) dependencies are
//! forbidden by `cargo xtask layers`. Like `lens_cmds`, this module processes
//! surfaces directly and plugs into smart-filter re-render through
//! [`apply_to_surface`].
//!
//! Backends: macOS uses Apple Vision (OS frameworks, zero weights); other
//! platforms degrade gracefully — skin/light work falls back to the chroma
//! skin rule and geometry is skipped until the ONNX decoders land.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use photocraft_beauty::geom::{
    BodyGeom, BodyWarpParams, FaceGeom, FaceWarpParams, Point, Primitive, StripSpec, body_warps, complete_body_joints, face_warps, segment_strip_mask,
    skin_chroma_score,
};
use photocraft_beauty::light::{adjust_saturation, brighten, enhance_eye_detail, sharpen};
use photocraft_beauty::mask::{convex_hull, feather_box, fill_convex_poly, resample_bilinear, subtract, union_max};
use photocraft_beauty::relight::{self, RelightSpec};
use photocraft_beauty::skin::{SmoothSpec, fade_blemishes, smooth_skin, uniform_tone};
use photocraft_beauty::warp::Field;
use photocraft_beauty::{BeautifyParams, Error as BeautyError};
use photocraft_color::ColorMode;
use photocraft_doc::{Document, LayerContent, LayerId, SmartFilter};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::photo_cmds::Stopwatch;
use crate::{EngineError, Result, Session};

pub const PORTRAIT: &str = "filter.beautify.portrait";
pub const SKIN: &str = "filter.beautify.skin";
pub const RESHAPE: &str = "filter.beautify.reshape";

/// Covers 45 MP camera originals; keep a bounded allocation budget for the f32 pipeline.
const MAX_PIXELS: usize = 50_000_000;

/// Detection cache capacity (entries); evicted wholesale when full. Photos are
/// processed one at a time, so a tiny cache covers slider scrubbing.
const CACHE_CAP: usize = 8;

// ---------- params ----------

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn num(p: &Value, k: &str) -> Option<f32> {
    p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).map(|v| v as f32)
}

fn s01_opt(p: &Value, k: &str) -> Option<f32> {
    num(p, k).map(|v| v.clamp(0.0, 100.0))
}

/// Parses dialog units into [`BeautifyParams`]: an optional `preset` seeds the
/// values, then every present slider overrides. Unknown keys are ignored so
/// old smart-filter records keep decoding.
pub fn beauty_params(cmd: &str, p: &Value) -> Result<BeautifyParams> {
    let mut out = match p.get("preset").and_then(Value::as_str) {
        None | Some("") | Some("Manual") => BeautifyParams::default(),
        Some(name) => photocraft_beauty::params::builtin(name).map_err(|e| bad(cmd, format!("unknown preset `{name}` ({e})")))?,
    };
    if let Some(v) = s01_opt(p, "skinSmooth") {
        out.skin_smooth = v;
    }
    if let Some(v) = s01_opt(p, "skinTone") {
        out.skin_tone_uniform = v;
    }
    if let Some(v) = s01_opt(p, "blemishFade") {
        out.blemish_fade = v;
    }
    if let Some(v) = s01_opt(p, "textureKeep") {
        out.texture_keep = v;
    }
    if let Some(v) = s01_opt(p, "nasolabialFade") {
        out.nasolabial_fade = v;
    }
    if let Some(v) = s01_opt(p, "eyeEnlarge") {
        out.eye_enlarge = v;
    }
    if let Some(v) = s01_opt(p, "jawSlim") {
        out.jaw_slim = v;
    }
    if let Some(v) = s01_opt(p, "noseRefine") {
        out.nose_refine = v;
    }
    if let Some(v) = s01_opt(p, "noseContour") {
        out.nose_contour = v;
    }
    if let Some(v) = s01_opt(p, "subjectBrighten") {
        out.subject_brighten = v;
    }
    if let Some(v) = s01_opt(p, "virtualLight") {
        out.virtual_light = v;
    }
    if let Some(v) = s01_opt(p, "lightX") {
        out.light_x = v;
    }
    if let Some(v) = s01_opt(p, "lightY") {
        out.light_y = v;
    }
    if let Some(v) = s01_opt(p, "lightSize") {
        out.light_size = v;
    }
    if let Some(v) = s01_opt(p, "vline") {
        out.vline = v;
    }
    if let Some(v) = s01_opt(p, "mouthLift") {
        out.mouth_lift = v;
    }
    if let Some(v) = s01_opt(p, "cheekLift") {
        out.cheek_lift = v;
    }
    if let Some(v) = s01_opt(p, "slimBody") {
        out.slim_body = v;
    }
    if let Some(v) = s01_opt(p, "waistSlim") {
        out.waist_slim = v;
    }
    if let Some(v) = s01_opt(p, "slimLegs") {
        out.slim_legs = v;
    }
    if let Some(v) = s01_opt(p, "legLengthen") {
        out.leg_lengthen = v;
    }
    if let Some(v) = s01_opt(p, "headRatio") {
        out.head_ratio = v;
    }
    if let Some(v) = s01_opt(p, "bustSize") {
        out.bust_size = v;
    }
    if let Some(v) = s01_opt(p, "bustLift") {
        out.bust_lift = v;
    }
    if let Some(v) = s01_opt(p, "cleavage") {
        out.cleavage = v;
    }
    if let Some(v) = s01_opt(p, "hipCurve") {
        out.hip_curve = v;
    }
    if let Some(v) = s01_opt(p, "brighten") {
        out.brighten = v;
    }
    if let Some(v) = num(p, "saturation") {
        out.saturation = v.clamp(-100.0, 100.0);
    }
    if let Some(v) = s01_opt(p, "eyeDetail") {
        out.eye_detail = v;
    }
    if let Some(v) = s01_opt(p, "sharpen") {
        out.sharpen = v;
    }
    Ok(out.normalized())
}

// ---------- detection cache ----------

/// Owned person mask (backend-agnostic copy for the cache).
#[derive(Clone)]
struct CachedMask {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

/// One cached detection pass over a photo's bytes.
#[derive(Clone, Default)]
struct CachedDetection {
    faces: Vec<FaceGeom>,
    bodies: Vec<BodyGeom>,
    person_mask: Option<CachedMask>,
}

static DETECTION_CACHE: LazyLock<Mutex<HashMap<[u8; 32], CachedDetection>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn cache_lookup(key: &[u8; 32]) -> Option<CachedDetection> {
    DETECTION_CACHE.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
}

fn cache_store(key: [u8; 32], det: CachedDetection) {
    let mut cache = DETECTION_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, det);
}

#[cfg(target_vendor = "apple")]
fn detect_backend(rgb8: &[u8], w: u32, h: u32) -> CachedDetection {
    let mut out = CachedDetection::default();
    // Detection failures degrade to the chroma fallback instead of erroring:
    // a filter must never refuse to run because Vision had a bad day.
    let Ok(mut det) = photocraft_beauty::apple::detect_rgb8(rgb8, w, h) else {
        return out;
    };
    // Vision returns a near-zero garbage mask when no person is present; only
    // a mask with real mass counts (a beautified subject fills >> 1%).
    if let Some(m) = &det.person_mask {
        let mean = m.data.iter().sum::<f32>() / m.data.len().max(1) as f32;
        if mean.is_nan() || mean <= 0.01 {
            det.person_mask = None;
        }
    }
    out.faces = det.faces;
    out.bodies = det.bodies;
    out.person_mask = det.person_mask.map(|m| CachedMask { w: m.w, h: m.h, data: m.data });
    out
}

#[cfg(not(target_vendor = "apple"))]
fn detect_backend(_rgb8: &[u8], _w: u32, _h: u32) -> CachedDetection {
    // ONNX backends (YuNet/MoveNet) land here; until then the chroma fallback.
    CachedDetection::default()
}

/// Detects faces/bodies/masks for RGB8 bytes, cached by content hash.
/// Returns the detection plus whether it came from the cache.
fn detect_cached(rgb8: &[u8], w: u32, h: u32) -> (CachedDetection, bool) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&w.to_le_bytes());
    hasher.update(&h.to_le_bytes());
    hasher.update(rgb8);
    let key: [u8; 32] = hasher.finalize().into();
    if let Some(hit) = cache_lookup(&key) {
        return (hit, true);
    }
    let det = detect_backend(rgb8, w, h);
    cache_store(key, det.clone());
    (det, false)
}

// ---------- masks ----------

fn beauty_err(cmd: &str, e: BeautyError) -> EngineError {
    EngineError::Other(format!("{cmd}: beautify failed ({e})"))
}

fn to_pixels(points: &[Point], w: usize, h: usize) -> Vec<[f32; 2]> {
    points.iter().map(|p| [p.x * w as f32, p.y * h as f32]).collect()
}

/// Union face-skin mask: oval hulls minus organ cutouts, feathered.
fn face_skin_mask(faces: &[FaceGeom], chroma: &[f32], w: usize, h: usize, cmd: &str) -> Result<Vec<f32>> {
    let mut acc = vec![0.0f32; w * h];
    for face in faces {
        let regions = face.skin_regions().map_err(|e| beauty_err(cmd, e))?;
        let hull = convex_hull(&to_pixels(&regions.oval, w, h));
        if hull.len() < 3 {
            continue;
        }
        let mut m = fill_convex_poly(w, h, &hull).map_err(|e| beauty_err(cmd, e))?;
        for cut in &regions.cutouts {
            let c = fill_convex_poly(w, h, &to_pixels(cut, w, h)).map_err(|e| beauty_err(cmd, e))?;
            m = subtract(&m, &c, w, h).map_err(|e| beauty_err(cmd, e))?;
        }
        acc = union_max(&acc, &m, w, h).map_err(|e| beauty_err(cmd, e))?;
    }
    // Only skin-coloured pixels inside the oval: hair, brows and the background inside the hull
    // are not skin and must not be smoothed, brightened or re-toned with the face.
    for (a, c) in acc.iter_mut().zip(chroma.iter()) {
        *a *= c;
    }
    let feather = (w.min(h) / 200).clamp(1, 6);
    feather_box(&acc, w, h, feather).map_err(|e| beauty_err(cmd, e))
}

/// Soft eye discs for the detail pass.
fn eye_mask(faces: &[FaceGeom], w: usize, h: usize, cmd: &str) -> Result<Vec<f32>> {
    let mut acc = vec![0.0f32; w * h];
    let min_dim = w.min(h) as f32;
    for face in faces {
        let r = (face.eye_radius * min_dim * 2.0).clamp(3.0, min_dim * 0.2);
        for eye in [face.left_eye, face.right_eye] {
            let spec =
                StripSpec { ax: eye.x * w as f32, ay: eye.y * h as f32, bx: eye.x * w as f32, by: eye.y * h as f32, radius: r, feather: (r * 0.6).max(1.0) };
            let m = segment_strip_mask(w, h, &spec).map_err(|e| beauty_err(cmd, e))?;
            acc = union_max(&acc, &m, w, h).map_err(|e| beauty_err(cmd, e))?;
        }
    }
    Ok(acc)
}

/// Nasolabial-fold strips for localized smoothing.
fn fold_mask(faces: &[FaceGeom], w: usize, h: usize, cmd: &str) -> Result<Vec<f32>> {
    let mut acc = vec![0.0f32; w * h];
    let min_dim = w.min(h) as f32;
    for face in faces {
        let Ok((_, folds)) = face.cheek_regions() else {
            continue;
        };
        let eye_dist = face.left_eye.dist(face.right_eye).max(1e-4) * min_dim;
        let r = (eye_dist * 0.10).clamp(2.0, min_dim * 0.1);
        for (a, b) in &folds {
            let spec = StripSpec { ax: a.x * w as f32, ay: a.y * h as f32, bx: b.x * w as f32, by: b.y * h as f32, radius: r, feather: r.max(1.0) };
            let m = segment_strip_mask(w, h, &spec).map_err(|e| beauty_err(cmd, e))?;
            acc = union_max(&acc, &m, w, h).map_err(|e| beauty_err(cmd, e))?;
        }
    }
    Ok(acc)
}

/// Face-oval union (no cutouts): the gate for face warps so the background
/// and organs stay put while the jawline moves.
fn face_gate_mask(faces: &[FaceGeom], w: usize, h: usize, cmd: &str) -> Result<Vec<f32>> {
    let mut acc = vec![0.0f32; w * h];
    for face in faces {
        let regions = face.skin_regions().map_err(|e| beauty_err(cmd, e))?;
        let hull = convex_hull(&to_pixels(&regions.oval, w, h));
        if hull.len() < 3 {
            continue;
        }
        let m = fill_convex_poly(w, h, &hull).map_err(|e| beauty_err(cmd, e))?;
        acc = union_max(&acc, &m, w, h).map_err(|e| beauty_err(cmd, e))?;
    }
    let feather = (w.min(h) / 120).clamp(2, 10);
    feather_box(&acc, w, h, feather).map_err(|e| beauty_err(cmd, e))
}

/// Approximate person mask from joints when segmentation is unavailable:
/// torso + limb strips plus the face ovals. Generous on purpose (it only
/// gates warps, and missing coverage would freeze a limb mid-warp).
fn approx_person_mask(bodies: &[BodyGeom], faces: &[FaceGeom], w: usize, h: usize, cmd: &str) -> Result<Vec<f32>> {
    let mut acc = face_gate_mask(faces, w, h, cmd)?;
    let min_dim = w.min(h) as f32;
    let mut strip = |a: Option<Point>, b: Option<Point>, radius: f32| -> Result<()> {
        let (Some(a), Some(b)) = (a, b) else {
            return Ok(());
        };
        if !(a.is_valid() && b.is_valid()) {
            return Ok(());
        }
        let spec = StripSpec {
            ax: a.x * w as f32,
            ay: a.y * h as f32,
            bx: b.x * w as f32,
            by: b.y * h as f32,
            radius: radius.max(2.0),
            feather: (radius * 0.5).max(1.0),
        };
        let m = segment_strip_mask(w, h, &spec).map_err(|e| beauty_err(cmd, e))?;
        acc = union_max(&acc, &m, w, h).map_err(|e| beauty_err(cmd, e))?;
        Ok(())
    };
    for body in bodies {
        let j = &body.joints;
        let shoulder_w = j.left_shoulder.zip(j.right_shoulder).map(|(a, b)| a.dist(b)).unwrap_or(0.15) * min_dim;
        let r = (shoulder_w * 0.30).clamp(6.0, min_dim * 0.2);
        strip(j.left_shoulder, j.left_hip, r)?;
        strip(j.right_shoulder, j.right_hip, r)?;
        strip(j.left_shoulder, j.right_shoulder, r * 0.8)?;
        strip(j.left_hip, j.right_hip, r)?;
        let leg_r = (r * 0.55).max(4.0);
        strip(j.left_hip, j.left_knee, leg_r)?;
        strip(j.left_knee, j.left_ankle, leg_r * 0.8)?;
        strip(j.right_hip, j.right_knee, leg_r)?;
        strip(j.right_knee, j.right_ankle, leg_r * 0.8)?;
    }
    feather_box(&acc, w, h, 2).map_err(|e| beauty_err(cmd, e))
}

// ---------- pipeline ----------

/// Which half of the params a command applies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Skin,
    Reshape,
    Full,
}

fn scope_of(id: &str) -> Option<Scope> {
    match id {
        SKIN => Some(Scope::Skin),
        RESHAPE => Some(Scope::Reshape),
        PORTRAIT => Some(Scope::Full),
        _ => None,
    }
}

fn face_warp_params(p: &BeautifyParams) -> FaceWarpParams {
    FaceWarpParams {
        eye_enlarge: p.eye_enlarge,
        jaw_slim: p.jaw_slim,
        nose_refine: p.nose_refine,
        vline: p.vline,
        mouth_lift: p.mouth_lift,
        cheek_lift: p.cheek_lift,
    }
}

fn body_warp_params(p: &BeautifyParams) -> BodyWarpParams {
    BodyWarpParams {
        slim_body: p.slim_body,
        waist_slim: p.waist_slim,
        slim_legs: p.slim_legs,
        leg_lengthen: p.leg_lengthen,
        head_ratio: p.head_ratio,
        bust_size: p.bust_size,
        bust_lift: p.bust_lift,
        cleavage: p.cleavage,
        hip_curve: p.hip_curve,
    }
}

/// Interleaved RGB8 (for detection) from normalized f32 samples.
fn rgb8_from_f32(px: &[f32], ch: usize) -> Vec<u8> {
    let pixels = px.len() / ch.max(1);
    let mut out = Vec::with_capacity(pixels * 3);
    for i in 0..pixels {
        if ch >= 3 {
            for c in 0..3 {
                out.push((px[i * ch + c].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        } else {
            let v = (px[i * ch].clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}

struct WorkMasks {
    skin: Vec<f32>,
    eyes: Vec<f32>,
    folds: Vec<f32>,
    person: Option<Vec<f32>>,
    face_gate: Vec<f32>,
}

/// Builds every working mask for the rect (pre-warp geometry).
fn build_masks(cmd: &str, det: &CachedDetection, img: &[f32], w: usize, h: usize, ch: usize, want_folds: bool) -> Result<WorkMasks> {
    // Soft skin-likeness: the face oval is intersected with it, so hair, brows and the background
    // inside the hull are not treated as face skin, and the body skin is it gated by the person
    // mask. (The score has smooth edges: a hard 0/1 rule leaves holes on shadowed skin.)
    let chroma = skin_chroma_score(img, w, h, ch).map_err(|e| beauty_err(cmd, e))?;
    let face_skin = face_skin_mask(&det.faces, &chroma, w, h, cmd)?;
    let eyes = eye_mask(&det.faces, w, h, cmd)?;
    let folds = if want_folds { fold_mask(&det.faces, w, h, cmd)? } else { vec![0.0; w * h] };
    let face_gate = face_gate_mask(&det.faces, w, h, cmd)?;
    // Person mask: resampled segmentation, else a joints-based approximation.
    let mut person = match &det.person_mask {
        Some(m) => Some(resample_bilinear(&m.data, m.w, m.h, w, h).map_err(|e| beauty_err(cmd, e))?),
        None if !det.bodies.is_empty() => Some(approx_person_mask(&det.bodies, &det.faces, w, h, cmd)?),
        None => None,
    };
    // Discard near-empty masks (backend noise floor on person-less photos):
    // a near-zero person mask would multiply all body skin down to nothing.
    if let Some(p) = &person {
        let mean = p.iter().sum::<f32>() / p.len().max(1) as f32;
        if mean.is_nan() || mean <= 0.01 {
            person = None;
        }
    }
    let body_skin = match &person {
        Some(p) => chroma.iter().zip(p.iter()).map(|(c, m)| c * m).collect(),
        None if det.faces.is_empty() => chroma.clone(),
        None => vec![0.0; w * h],
    };
    let skin = union_max(&face_skin, &body_skin, w, h).map_err(|e| beauty_err(cmd, e))?;
    Ok(WorkMasks { skin, eyes, folds, person, face_gate })
}

/// Neck reference for mask-based torso estimation: the associated face's chin
/// when one lines up with the body, else below the nose, else from the bbox.
fn neck_for_body(faces: &[FaceGeom], body: &BodyGeom) -> Point {
    let cx = (body.bbox[0] + body.bbox[2]) * 0.5;
    let mut best: Option<&FaceGeom> = None;
    let mut best_dx = 0.25f32;
    for face in faces {
        let dx = (face.chin.x - cx).abs();
        if dx < best_dx {
            best_dx = dx;
            best = Some(face);
        }
    }
    if let Some(face) = best {
        return face.chin;
    }
    if let Some(nose) = body.joints.nose
        && nose.is_valid()
    {
        return Point { x: nose.x, y: (nose.y + 0.06).clamp(0.0, 1.0) };
    }
    Point { x: cx.clamp(0.0, 1.0), y: (body.bbox[1] + (body.bbox[3] - body.bbox[1]) * 0.15).clamp(0.0, 1.0) }
}
/// Protect the entire head (not merely skin pixels) from torso/leg deformation.
/// The plateau includes the face box; feathering happens outside it, toward the neck.
fn body_only_gate(det: &CachedDetection, person: Option<&[f32]>, w: usize, h: usize) -> Vec<f32> {
    let mut gate = person.map_or_else(|| vec![1.0; w * h], <[f32]>::to_vec);
    let mut boxes: Vec<[f32; 4]> = det.faces.iter().map(|f| f.bbox).collect();
    // Even if face detection fails, keep the pose-estimated head off-limits.
    for body in &det.bodies {
        if let Some(nose) = body.joints.nose.filter(|p| p.is_valid()) {
            let near_face = det.faces.iter().any(|f| nose.x >= f.bbox[0] && nose.x <= f.bbox[2] && nose.y >= f.bbox[1] && nose.y <= f.bbox[3]);
            if !near_face {
                let half = (body.bbox[2] - body.bbox[0]).abs() * 0.22;
                boxes.push([nose.x - half, nose.y - half, nose.x + half, nose.y + half * 1.25]);
            }
        }
    }
    for [x0, y0, x1, y1] in boxes {
        let fw = (x1 - x0).max(0.001);
        let fh = (y1 - y0).max(0.001);
        let cx = (x0 + x1) * 0.5;
        let cy = y0 + fh * 0.35;
        for (i, m) in gate.iter_mut().enumerate() {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            let r = ((x - cx) / (fw * 0.78)).hypot((y - cy) / (fh * 0.95));
            let t = ((r - 1.0) / 0.25).clamp(0.0, 1.0);
            *m *= t * t * (3.0 - 2.0 * t);
        }
    }
    gate
}

/// Applies geometry warps to the image and its masks. Returns the warped
/// image (or a copy when there is nothing to warp).
#[allow(clippy::too_many_arguments)]
fn apply_warps(
    cmd: &str,
    params: &BeautifyParams,
    det: &CachedDetection,
    masks: &mut WorkMasks,
    img: &[f32],
    w: usize,
    h: usize,
    ch: usize,
) -> Result<Vec<f32>> {
    let fw = face_warp_params(params);
    let bw = body_warp_params(params);
    let want_face = fw.eye_enlarge > 0.0 || fw.jaw_slim > 0.0 || fw.nose_refine > 0.0 || fw.vline > 0.0 || fw.mouth_lift > 0.0 || fw.cheek_lift > 0.0;
    let want_body = bw.slim_body > 0.0
        || bw.waist_slim > 0.0
        || bw.slim_legs > 0.0
        || bw.leg_lengthen > 0.0
        || bw.head_ratio > 0.0
        || bw.bust_size > 0.0
        || bw.bust_lift > 0.0
        || bw.cleavage > 0.0
        || bw.hip_curve > 0.0;
    if (!want_face && !want_body) || (det.faces.is_empty() && det.bodies.is_empty()) {
        return Ok(img.to_vec());
    }
    // Face and body warps are built into separate fields: the face one gets a saturating limit
    // (Liquify's brush saturates the same way, so overlapping jaw strokes narrow the face instead
    // of stacking into a cone), while body work keeps its full range (leg lengthening is large).
    let mut face_field = Field::new(w, h).map_err(|e| beauty_err(cmd, e))?;
    if want_face {
        for face in &det.faces {
            let prims: Vec<Primitive> = face_warps(face, w, h, &fw).map_err(|e| beauty_err(cmd, e))?;
            for prim in &prims {
                prim.apply(&mut face_field, Some(&masks.face_gate)).map_err(|e| beauty_err(cmd, e))?;
            }
        }
        face_field.soft_clamp(w.min(h) as f32 * 0.020);
    }
    let mut body_field = Field::new(w, h).map_err(|e| beauty_err(cmd, e))?;
    if want_body {
        let body_gate = body_only_gate(det, masks.person.as_deref(), w, h);
        for body in &det.bodies {
            // Complete joints from the silhouette (pose backends often miss
            // the legs); lengthening only with a split silhouette or full
            // detections, slimming also on the unsplit fallback.
            let neck = neck_for_body(&det.faces, body);
            let (joints, legs_split) = complete_body_joints(&body.joints, masks.person.as_deref(), w, h, neck).map_err(|e| beauty_err(cmd, e))?;
            let mut bw = bw;
            // Head proportion is explicit and independent; body slimming never includes it.
            let head_ratio = bw.head_ratio;
            bw.head_ratio = 0.0;
            if !legs_split {
                bw.leg_lengthen = 0.0;
            }
            let completed = BodyGeom { bbox: body.bbox, joints };
            let prims = body_warps(&completed, w, h, &bw).map_err(|e| beauty_err(cmd, e))?;
            for prim in &prims {
                prim.apply(&mut body_field, Some(&body_gate)).map_err(|e| beauty_err(cmd, e))?;
            }
            if head_ratio > 0.0 {
                let head = BodyWarpParams { head_ratio, ..BodyWarpParams::default() };
                for prim in body_warps(&completed, w, h, &head).map_err(|e| beauty_err(cmd, e))? {
                    prim.apply(&mut body_field, masks.person.as_deref()).map_err(|e| beauty_err(cmd, e))?;
                }
            }
        }
    }
    let mut field = face_field;
    field.add_from(&body_field).map_err(|e| beauty_err(cmd, e))?;
    if field.max_abs() == 0.0 {
        return Ok(img.to_vec());
    }
    // Warp the masks with the same field so skin work lands on the moved face.
    for mask in [&mut masks.skin, &mut masks.eyes, &mut masks.folds] {
        let warped = field.remap(mask, 1).map_err(|e| beauty_err(cmd, e))?;
        *mask = warped;
    }
    if let Some(person) = &mut masks.person {
        *person = field.remap(person, 1).map_err(|e| beauty_err(cmd, e))?;
    }
    field.remap(img, ch).map_err(|e| beauty_err(cmd, e))
}

/// Applies skin + light work to the (possibly warped) image.
#[allow(clippy::too_many_arguments)]
fn apply_skin_light(cmd: &str, params: &BeautifyParams, masks: &WorkMasks, faces: &[FaceGeom], img: &[f32], w: usize, h: usize, ch: usize) -> Result<Vec<f32>> {
    let min_dim = w.min(h);
    let smooth_r = (min_dim / 120).clamp(2, 10);
    let mut cur = img.to_vec();
    if params.skin_smooth > 0.0 {
        let spec = SmoothSpec { strength: params.skin_smooth / 100.0, texture_keep: params.texture_keep / 100.0, radius: smooth_r, eps: 0.001 };
        cur = smooth_skin(&cur, w, h, ch, &masks.skin, &spec).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.skin_tone_uniform > 0.0 {
        cur = uniform_tone(&cur, w, h, ch, &masks.skin, params.skin_tone_uniform / 100.0, (smooth_r * 2).clamp(4, 24)).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.blemish_fade > 0.0 {
        cur = fade_blemishes(&cur, w, h, ch, &masks.skin, params.blemish_fade / 100.0).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.nasolabial_fade > 0.0 {
        let spec = SmoothSpec { strength: params.nasolabial_fade / 100.0, texture_keep: 0.5, radius: (smooth_r / 2).max(1), eps: 0.001 };
        cur = smooth_skin(&cur, w, h, ch, &masks.folds, &spec).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.brighten > 0.0 {
        cur = brighten(&cur, w, h, ch, &masks.skin, params.brighten / 100.0).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.saturation != 0.0 {
        cur = adjust_saturation(&cur, w, h, ch, &masks.skin, params.saturation / 100.0).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.eye_detail > 0.0 {
        cur = enhance_eye_detail(&cur, w, h, ch, &masks.eyes, params.eye_detail / 100.0).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.sharpen > 0.0 {
        cur = sharpen(&cur, w, h, ch, &masks.skin, params.sharpen / 100.0, 2).map_err(|e| beauty_err(cmd, e))?;
    }
    if params.nose_contour > 0.0 || params.subject_brighten > 0.0 || params.virtual_light > 0.0 {
        let spec = RelightSpec {
            nose_contour: params.nose_contour / 100.0,
            subject_brighten: params.subject_brighten / 100.0,
            virtual_light: params.virtual_light / 100.0,
            light_x: params.light_x / 100.0,
            light_y: params.light_y / 100.0,
            light_size: params.light_size / 100.0,
        };
        let person = masks.person.as_deref().unwrap_or(&masks.face_gate);
        cur = relight::apply(&cur, w, h, ch, person, &masks.skin, faces, &spec).map_err(|e| beauty_err(cmd, e))?;
    }
    Ok(cur)
}

struct BeautifyOutcome {
    surface: Surface,
    faces: usize,
    bodies: usize,
    cache_hit: bool,
}

/// Runs the full pipeline on `surf` clipped to `canvas`. Pure in the surface
/// sense: the input is never modified.
fn beautify_surface(id: &str, params: &BeautifyParams, surf: &Surface, canvas: Rect) -> Result<BeautifyOutcome> {
    let scope = scope_of(id).ok_or_else(|| EngineError::Other(format!("unknown beautify command {id}")))?;
    let work = surf.content_bounds().intersect(&canvas);
    if work.is_empty() {
        return Ok(BeautifyOutcome { surface: surf.clone(), faces: 0, bodies: 0, cache_hit: false });
    }
    let (w, h) = (work.width() as usize, work.height() as usize);
    let pixels = w.checked_mul(h).ok_or_else(|| bad(id, "work area too large"))?;
    if pixels > MAX_PIXELS {
        return Err(bad(id, format!("work area {w}x{h} is too large for beautify (>{MAX_PIXELS} px)")));
    }
    let ch = surf.channels().max(1);
    let img = surf.read_region(work);
    if img.len() != pixels * ch {
        return Err(EngineError::Other(format!("{id}: short read ({} of {})", img.len(), pixels * ch)));
    }
    let rgb8 = rgb8_from_f32(&img, ch);
    let (det, cache_hit) = detect_cached(&rgb8, w as u32, h as u32);
    let faces = det.faces.len();
    let bodies = det.bodies.len();

    let do_geom = matches!(scope, Scope::Reshape | Scope::Full) && params.has_geometry();
    let do_skin = (matches!(scope, Scope::Skin | Scope::Full) && (params.has_skin_work() || params.has_light_work()))
        || (matches!(scope, Scope::Reshape) && params.nose_contour > 0.0);
    if !do_geom && !do_skin {
        return Ok(BeautifyOutcome { surface: surf.clone(), faces, bodies, cache_hit });
    }
    let mut masks = build_masks(id, &det, &img, w, h, ch, params.nasolabial_fade > 0.0 && do_skin)?;
    let mut cur = if do_geom { apply_warps(id, params, &det, &mut masks, &img, w, h, ch)? } else { img };
    if do_skin {
        let shape_only = BeautifyParams { nose_contour: params.nose_contour, ..BeautifyParams::default() };
        let light_params = if matches!(scope, Scope::Reshape) { &shape_only } else { params };
        cur = apply_skin_light(id, light_params, &masks, &det.faces, &cur, w, h, ch)?;
    }
    let mut out = surf.clone();
    out.write_region(work, &cur);
    out.prune();
    Ok(BeautifyOutcome { surface: out, faces, bodies, cache_hit })
}

/// Smart-filter hook: re-applies a stored filter of this module (`None` for
/// other ids, or when the area is too large to re-render).
pub fn apply_to_surface(id: &str, params: &Value, surf: &Surface, canvas: Rect) -> Option<Surface> {
    scope_of(id)?;
    let parsed = beauty_params(id, params).ok()?;
    beautify_surface(id, &parsed, surf, canvas).map(|o| o.surface).ok()
}

// ---------- commands ----------

fn target(s: &Session, p: &Value) -> Result<LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(v) => Ok(LayerId(v)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    match &d.doc.layer(id).ok_or("no active layer")?.content {
        LayerContent::Raster(_) => {}
        LayerContent::Smart(sm) if sm.cache.is_some() => {}
        other => return Err(format!("filters need a pixel layer (active layer is {} {} layer)", other.article(), other.kind_name())),
    }
    match d.doc.mode {
        ColorMode::Rgb | ColorMode::Grayscale => Ok(()),
        m => Err(format!("Beautify needs an RGB or Grayscale document (this one is {m:?})")),
    }
}

/// Mixes `new` over `old` by the selection's coverage (straight-colour mix;
/// beautify never changes alpha, so no premultiplied path is needed).
fn mix_by_selection(old: &Surface, new: &Surface, sel: &Surface) -> Surface {
    let area = old.content_bounds().union(&new.content_bounds());
    if area.is_empty() {
        return new.clone();
    }
    let mut out = new.clone();
    let n = new.channels().max(1);
    let (o, w) = (old.read_region(area), new.read_region(area));
    if o.len() != w.len() {
        return out;
    }
    let aw = area.width() as usize;
    let mut r = vec![0.0f32; o.len()];
    for (i, ((op, np), rp)) in o.chunks_exact(n).zip(w.chunks_exact(n)).zip(r.chunks_exact_mut(n)).enumerate() {
        let k = sel.sample_channel(area.x0 + (i % aw) as i32, area.y0 + (i / aw) as i32, 0).clamp(0.0, 1.0);
        for c in 0..n {
            rp[c] = op[c] + (np[c] - op[c]) * k;
        }
    }
    out.write_region(area, &r);
    out.prune();
    out
}

fn run(id: &str, label: &str, s: &mut Session, p: &Value) -> Result<Value> {
    let params = beauty_params(id, p)?;
    let target_id = target(s, p)?;
    let mut stored = p.clone();
    if let Value::Object(m) = &mut stored {
        m.remove("layer");
    }
    let t0 = Stopwatch::start();
    let (faces, bodies, cache_hit) = s.edit(label, |doc: &mut Document, _| {
        let canvas = doc.bounds();
        let selection = doc.selection.clone();
        let locks = doc.effective_locks(target_id);
        let l = doc.layer_mut(target_id).ok_or(EngineError::NoLayer(target_id))?;
        if let LayerContent::Smart(_) = l.content {
            let sf = SmartFilter { command: id.to_string(), params: stored.clone(), blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
            return crate::smart_cmds::add_smart_filter(doc, target_id, sf, selection.as_ref()).map(|()| (0, 0, false));
        }
        if locks.all || locks.pixels {
            return Err(EngineError::Other(format!("layer \"{}\" is locked", l.name)));
        }
        let LayerContent::Raster(surf) = &mut l.content else {
            return Err(EngineError::Other(format!("{label} needs a pixel layer")));
        };
        let outcome = beautify_surface(id, &params, surf, canvas)?;
        let mixed = match &selection {
            Some(sel) => mix_by_selection(surf, &outcome.surface, sel),
            None => outcome.surface,
        };
        *surf = mixed;
        Ok((outcome.faces, outcome.bodies, outcome.cache_hit))
    })?;
    Ok(json!({
        "layer": target_id.0,
        "faces": faces,
        "bodies": bodies,
        "cacheHit": cache_hit,
        "ms": t0.ms(),
    }))
}

fn run_portrait(s: &mut Session, p: &Value) -> Result<Value> {
    run(PORTRAIT, "Portrait Beautify", s, p)
}

fn run_skin(s: &mut Session, p: &Value) -> Result<Value> {
    run(SKIN, "Portrait Skin", s, p)
}

fn run_reshape(s: &mut Session, p: &Value) -> Result<Value> {
    run(RESHAPE, "Portrait Reshape", s, p)
}

const SKIN_DOC: &str = r##"{"preset":"Manual|Natural|Cosplay|Editorial","skinSmooth":0..100=0,"skinTone":0..100=0,"blemishFade":0..100=0,"textureKeep":0..100=85,"nasolabialFade":0..100=0,"brighten":0..100=0,"subjectBrighten":0..100=0,"virtualLight":0..100=0,"lightX":0..100=30,"lightY":0..100=20,"lightSize":0..100=65,"saturation":-100..100=0,"eyeDetail":0..100=0,"sharpen":0..100=0}"##;
const RESHAPE_DOC: &str = r##"{"preset":"Manual|Natural|Cosplay|Editorial","eyeEnlarge":0..100=0,"jawSlim":0..100=0,"noseRefine":0..100=0,"noseContour":0..100=0,"vline":0..100=0,"mouthLift":0..100=0,"cheekLift":0..100=0,"slimBody":0..100=0,"waistSlim":0..100=0,"slimLegs":0..100=0,"legLengthen":0..100=0,"headRatio":0..100=0,"bustSize":0..100=0,"bustLift":0..100=0,"cleavage":0..100=0,"hipCurve":0..100=0}"##;
const PORTRAIT_DOC: &str = r##"{"preset":"Manual|Natural|Cosplay|Editorial","skinSmooth":0..100=0,"skinTone":0..100=0,"blemishFade":0..100=0,"textureKeep":0..100=85,"nasolabialFade":0..100=0,"eyeEnlarge":0..100=0,"jawSlim":0..100=0,"noseRefine":0..100=0,"noseContour":0..100=0,"vline":0..100=0,"mouthLift":0..100=0,"cheekLift":0..100=0,"slimBody":0..100=0,"waistSlim":0..100=0,"slimLegs":0..100=0,"legLengthen":0..100=0,"headRatio":0..100=0,"bustSize":0..100=0,"bustLift":0..100=0,"cleavage":0..100=0,"hipCurve":0..100=0,"brighten":0..100=0,"subjectBrighten":0..100=0,"virtualLight":0..100=0,"lightX":0..100=30,"lightY":0..100=20,"lightSize":0..100=65,"saturation":-100..100=0,"eyeDetail":0..100=0,"sharpen":0..100=0}"##;

/// The command specs of this module.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: PORTRAIT,
            label: "Portrait Beautify…",
            menu: &["Filter", "Beautify"],
            shortcut: None,
            params: PORTRAIT_DOC,
            enabled,
            run: run_portrait,
            journal: true,
        },
        CommandSpec {
            id: SKIN,
            label: "Portrait Skin…",
            menu: &["Filter", "Beautify"],
            shortcut: None,
            params: SKIN_DOC,
            enabled,
            run: run_skin,
            journal: true,
        },
        CommandSpec {
            id: RESHAPE,
            label: "Portrait Reshape…",
            menu: &["Filter", "Beautify"],
            shortcut: None,
            params: RESHAPE_DOC,
            enabled,
            run: run_reshape,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_slim_preserves_face_even_with_overlapping_torso() {
        let face = FaceGeom::from_sparse(
            [0.30, 0.10, 0.70, 0.40],
            Point { x: 0.40, y: 0.20 },
            Point { x: 0.60, y: 0.20 },
            Point { x: 0.50, y: 0.28 },
            Point { x: 0.44, y: 0.33 },
            Point { x: 0.56, y: 0.33 },
        )
        .unwrap();
        let body = BodyGeom {
            bbox: [0.2, 0.1, 0.8, 0.98],
            joints: photocraft_beauty::geom::Joints {
                nose: Some(Point { x: 0.5, y: 0.28 }),
                left_shoulder: Some(Point { x: 0.25, y: 0.30 }),
                right_shoulder: Some(Point { x: 0.75, y: 0.30 }),
                left_hip: Some(Point { x: 0.35, y: 0.60 }),
                right_hip: Some(Point { x: 0.65, y: 0.60 }),
                ..Default::default()
            },
        };
        let det = CachedDetection { faces: vec![face], bodies: vec![body], person_mask: None };
        let (w, h) = (100, 120);
        let img: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32 / 251.0).collect();
        let mut masks =
            WorkMasks { skin: vec![0.0; w * h], eyes: vec![0.0; w * h], folds: vec![0.0; w * h], person: Some(vec![1.0; w * h]), face_gate: vec![0.0; w * h] };
        let p = BeautifyParams { slim_body: 100.0, waist_slim: 100.0, ..Default::default() };
        let out = apply_warps(RESHAPE, &p, &det, &mut masks, &img, w, h, 3).unwrap();
        for y in 12..48 {
            for x in 30..70 {
                let i = (y * w + x) * 3;
                assert_eq!(&out[i..i + 3], &img[i..i + 3], "body work changed face at {x},{y}");
            }
        }
        assert_ne!(out, img, "body work must still affect the torso");
    }

    #[test]
    fn manual_preset_and_lighting_params_are_independent() {
        let p = beauty_params(
            PORTRAIT,
            &json!({"preset":"Manual","slimBody":55,"subjectBrighten":40,"virtualLight":25,"lightX":15,"lightY":20,"lightSize":70,"noseContour":30}),
        )
        .unwrap();
        assert_eq!(p.jaw_slim, 0.0);
        assert_eq!(p.skin_smooth, 0.0);
        assert_eq!(p.nose_refine, 0.0);
        assert_eq!(p.slim_body, 55.0);
        assert_eq!(p.subject_brighten, 40.0);
        assert_eq!(p.virtual_light, 25.0);
        assert_eq!(p.light_x, 15.0);
        assert_eq!(p.light_y, 20.0);
        assert_eq!(p.light_size, 70.0);
        assert_eq!(p.nose_contour, 30.0);
    }
    use photocraft_beauty::geom::Point;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 48, "height": 48})).unwrap();
        s.edit("skin", |doc, active| {
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let fmt = surf.format();
            // Skin-toned card (see the per-pixel cases below).
            for y in 0..48 {
                for x in 0..48 {
                    // Skin-toned card, a 3x3 dark blemish, and a sky-blue
                    // (non-skin) corner. Single-pixel dots are intentionally
                    // ignored as noise by the blemish opener.
                    let (r, g, b) = if x < 8 && y < 8 {
                        (0.35, 0.55, 0.85)
                    } else if (29..32).contains(&x) && (19..22).contains(&y) {
                        (0.35, 0.25, 0.20)
                    } else {
                        (0.85, 0.65, 0.55)
                    };
                    surf.write_pixel(x, y, &photocraft_raster::from_rgba(&fmt, [r, g, b, 1.0]));
                }
            }
            Ok(())
        })
        .unwrap();
        s
    }

    fn active_rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
    }

    #[test]
    fn params_preset_then_override() {
        let p = beauty_params(SKIN, &json!({"preset": "Natural"})).unwrap();
        assert_eq!(p.skin_smooth, 30.0);
        assert_eq!(p.texture_keep, 80.0);
        assert_eq!(p.eye_enlarge, 0.0);
        let p = beauty_params(SKIN, &json!({"preset": "Natural", "skinSmooth": 90, "saturation": 200})).unwrap();
        assert_eq!(p.skin_smooth, 90.0);
        assert_eq!(p.saturation, 100.0);
        assert!(beauty_params(SKIN, &json!({"preset": "Nope"})).is_err());
        // Non-finite input can never smuggle NaN into the pipeline.
        let p = beauty_params(SKIN, &json!({"skinSmooth": f64::NAN, "saturation": f64::INFINITY})).unwrap();
        assert_eq!(p.skin_smooth, 0.0);
        assert_eq!(p.saturation, 0.0);
    }

    #[test]
    fn rgb8_conversion_roundtrips_gray_and_rgb() {
        assert_eq!(rgb8_from_f32(&[0.0, 1.0, 0.5, 1.0], 2), vec![0, 0, 0, 128, 128, 128]);
        assert_eq!(rgb8_from_f32(&[1.0, 0.0, 0.5, 1.0], 4), vec![255, 0, 128]);
    }

    #[test]
    fn cache_returns_identical_detections() {
        let rgb = vec![128u8; 24 * 24 * 3];
        let (a, hit_a) = detect_cached(&rgb, 24, 24);
        let (b, hit_b) = detect_cached(&rgb, 24, 24);
        assert!(!hit_a);
        assert!(hit_b);
        assert_eq!(a.faces.len(), b.faces.len());
        assert_eq!(a.bodies.len(), b.bodies.len());
        assert_eq!(a.person_mask.is_some(), b.person_mask.is_some());
    }

    #[test]
    fn neck_prefers_aligned_face_chin() {
        use photocraft_beauty::geom::Joints;
        let face = FaceGeom::from_sparse(
            [0.3, 0.1, 0.7, 0.9],
            Point { x: 0.42, y: 0.38 },
            Point { x: 0.58, y: 0.38 },
            Point { x: 0.5, y: 0.55 },
            Point { x: 0.44, y: 0.70 },
            Point { x: 0.56, y: 0.70 },
        )
        .unwrap();
        let body = BodyGeom { bbox: [0.3, 0.2, 0.7, 0.98], joints: Joints::default() };
        assert_eq!(neck_for_body(&[face.clone()], &body), face.chin);
        // A far-away face belongs to someone else: bbox fallback.
        let far = FaceGeom::from_sparse(
            [0.8, 0.1, 0.95, 0.3],
            Point { x: 0.85, y: 0.15 },
            Point { x: 0.90, y: 0.15 },
            Point { x: 0.87, y: 0.20 },
            Point { x: 0.85, y: 0.25 },
            Point { x: 0.90, y: 0.25 },
        )
        .unwrap();
        let neck = neck_for_body(&[far], &body);
        assert!((neck.x - 0.5).abs() < 1e-6);
        assert!(neck.y > body.bbox[1] && neck.y < 0.5);
    }

    #[test]
    fn face_masks_cover_oval_not_eyes() {
        let face = FaceGeom::from_sparse(
            [0.3, 0.1, 0.7, 0.9],
            Point { x: 0.42, y: 0.38 },
            Point { x: 0.58, y: 0.38 },
            Point { x: 0.5, y: 0.55 },
            Point { x: 0.44, y: 0.70 },
            Point { x: 0.56, y: 0.70 },
        )
        .unwrap();
        // Skin score: skin everywhere except the top band (hair / background above the face).
        let mut chroma = vec![1.0f32; 100 * 100];
        for y in 0..40 {
            for x in 0..100 {
                chroma[y * 100 + x] = 0.0;
            }
        }
        let skin = face_skin_mask(&[face.clone()], &chroma, 100, 100, "test").unwrap();
        // Cheek is skin; the eye centre is cut out.
        assert!(skin[55 * 100 + 42] > 0.9, "cheek");
        assert!(skin[38 * 100 + 42] < 0.5, "left eye cutout");
        assert_eq!(skin[5 * 100 + 5], 0.0, "background");
        assert!(skin[20 * 100 + 50] < 0.2, "hair inside the oval is not face skin");
        let eyes = eye_mask(&[face], 100, 100, "test").unwrap();
        assert!(eyes[38 * 100 + 42] > 0.9, "eye disc");
        assert_eq!(eyes[5 * 100 + 5], 0.0);
    }

    #[test]
    fn identity_params_leave_pixels_bit_identical() {
        let mut s = session();
        assert!(s.is_enabled(PORTRAIT));
        let before_readings: Vec<[f32; 4]> = [(4, 4), (30, 20), (40, 40)].iter().map(|(x, y)| active_rgba(&s, *x, *y)).collect();
        let r = s.execute(PORTRAIT, json!({})).unwrap();
        assert!(r["ms"].as_f64().is_some());
        assert_eq!(r["faces"], json!(0));
        for (i, (x, y)) in [(4, 4), (30, 20), (40, 40)].iter().enumerate() {
            assert_eq!(active_rgba(&s, *x, *y), before_readings[i], "pixel ({x},{y})");
        }
        s.undo();
    }

    #[test]
    fn near_empty_person_mask_falls_back_to_chroma() {
        // Vision returns noise-floor masks for person-less photos; they must
        // not multiply the chroma fallback down to nothing.
        let mut det = CachedDetection::default();
        det.person_mask = Some(CachedMask { w: 4, h: 4, data: vec![0.001; 16] });
        let mut img = Vec::new();
        for _ in 0..8 * 8 {
            img.extend_from_slice(&[0.85, 0.65, 0.55, 1.0]);
        }
        let masks = build_masks("test", &det, &img, 8, 8, 4, false).unwrap();
        assert!(masks.person.is_none());
        assert!(masks.skin.iter().sum::<f32>() > 60.0);
    }

    #[test]
    fn blemish_fade_clears_dot_keeps_gray_corner() {
        let mut s = session();
        let corner_before = active_rgba(&s, 4, 4);
        // No faces in a flat card: the chroma fallback still finds the skin.
        let r = s.execute(SKIN, json!({"blemishFade": 100})).unwrap();
        assert_eq!(r["faces"], json!(0));
        let dot = active_rgba(&s, 30, 20);
        // The dark dot moved most of the way to the skin card.
        assert!(dot[0] > 0.70, "dot faded: {dot:?}");
        // The sky-blue corner is not skin: bit-identical.
        assert_eq!(active_rgba(&s, 4, 4), corner_before);
        s.undo();
        assert!(active_rgba(&s, 30, 20)[0] < 0.40);
    }

    #[test]
    fn reshape_without_detection_is_noop() {
        let mut s = session();
        let before = active_rgba(&s, 30, 20);
        // Geometry needs landmarks; with none found the layer is untouched.
        s.execute(RESHAPE, json!({"eyeEnlarge": 100, "slimBody": 100})).unwrap();
        assert_eq!(active_rgba(&s, 30, 20), before);
    }

    #[test]
    fn smart_filter_records_and_rerenders() {
        let mut s = session();
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        s.execute(SKIN, json!({"skinSmooth": 50})).unwrap();
        let d = s.active().unwrap();
        let filters = match &d.doc.layer(d.active_layer.unwrap()).unwrap().content {
            LayerContent::Smart(so) => so.smart_filters.clone(),
            _ => panic!("smart object expected"),
        };
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].command, SKIN);
        // Re-render path decodes the stored record.
        let surf = d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().clone();
        let canvas = d.doc.bounds();
        let out = apply_to_surface(SKIN, &filters[0].params, &surf, canvas);
        assert!(out.is_some());
        assert!(apply_to_surface("filter.blur.gaussianBlur", &json!({}), &surf, canvas).is_none());
    }
}
