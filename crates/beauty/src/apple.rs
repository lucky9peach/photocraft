//! Apple Vision detection backend (macOS only).
//!
//! Faces (+76-point landmarks), body pose (17 joints) and person segmentation
//! via the OS frameworks: no weights to ship, no licenses to clear, no extra
//! megabytes. This module only compiles on Apple targets; other platforms use
//! the ONNX backends (see [`crate::detect`]).
//!
//! The `apple-vision` bridge takes file paths, so detection writes one temp
//! PNG per call (detection runs once per photo, not per slider tick) and
//! deletes it afterwards. All coordinates convert here from Vision's
//! bottom-left origin to top-left normalized image coordinates.

use std::sync::atomic::{AtomicU64, Ordering};

use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};

use crate::Error;
use crate::geom::{BodyGeom, FaceGeom, Joints, Point};

/// Grayscale person mask, top-left origin, row-major 0..1.
#[derive(Clone, Debug, PartialEq)]
pub struct PersonMask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// One detection pass over a photo.
#[derive(Clone, Debug, Default)]
pub struct AppleDetection {
    pub faces: Vec<FaceGeom>,
    pub bodies: Vec<BodyGeom>,
    /// Union person mask (all instances), if segmentation succeeded.
    pub person_mask: Option<PersonMask>,
    /// Per-instance masks (pixel values select the instance).
    pub instances: Vec<PersonMask>,
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_png_path() -> std::path::PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    std::env::temp_dir().join(format!("portraitlab-vision-{}-{nanos}-{n}.png", std::process::id()))
}

fn encode_png_rgb8(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>, Error> {
    if w == 0 || h == 0 {
        return Err(Error::EmptyInput("apple png dimensions"));
    }
    if rgb.len() != w as usize * h as usize * 3 {
        return Err(Error::ShapeMismatch { what: "apple png input", expected: w as usize * h as usize * 3, found: rgb.len() });
    }
    let mut buf = Vec::new();
    PngEncoder::new(&mut buf).write_image(rgb, w, h, ColorType::Rgb8.into()).map_err(|e| Error::InvalidParam(format!("png encode: {e}")))?;
    Ok(buf)
}

fn vision_err(e: impl core::fmt::Display) -> Error {
    Error::InvalidParam(format!("vision request: {e}"))
}

/// Detects faces, bodies and person masks in RGB8 pixels.
pub fn detect_rgb8(rgb: &[u8], w: u32, h: u32) -> Result<AppleDetection, Error> {
    let png = encode_png_rgb8(rgb, w, h)?;
    let path = temp_png_path();
    std::fs::write(&path, &png).map_err(|e| Error::InvalidParam(format!("temp png write: {e}")))?;
    let result = detect_path(&path);
    let _ = std::fs::remove_file(&path);
    result
}

fn detect_path(path: &std::path::Path) -> Result<AppleDetection, Error> {
    use apple_vision::body_pose::detect_human_body_pose_in_path;
    use apple_vision::face_landmarks::detect_face_landmarks_in_path;
    use apple_vision::segmentation::{SegmentationQuality, generate_person_segmentation_in_path};

    let mut out = AppleDetection::default();
    // Faces are the must-have; bodies/masks degrade gracefully.
    let faces = detect_face_landmarks_in_path(path).map_err(vision_err)?;
    for face in faces {
        if let Some(geom) = map_face(&face) {
            out.faces.push(geom);
        }
    }
    if let Ok(poses) = detect_human_body_pose_in_path(path) {
        for pose in poses {
            out.bodies.push(map_body(&pose));
        }
    }
    if let Ok(Some(mask)) = generate_person_segmentation_in_path(path, SegmentationQuality::Balanced)
        && let Some(person) = map_mask(&mask)
    {
        out.person_mask = Some(person);
    }
    Ok(out)
}

/// Vision bbox (bottom-left) -> normalized top-left [x0, y0, x1, y1].
fn flip_bbox(x: f64, y: f64, w: f64, h: f64) -> [f32; 4] {
    [x as f32, (1.0 - y - h) as f32, (x + w) as f32, (1.0 - y) as f32]
}

fn centroid(pts: &[Point]) -> Option<Point> {
    if pts.is_empty() {
        return None;
    }
    let (sx, sy) = pts.iter().fold((0.0f64, 0.0f64), |(ax, ay), p| (ax + f64::from(p.x), ay + f64::from(p.y)));
    let n = pts.len() as f64;
    Some(Point { x: (sx / n) as f32, y: (sy / n) as f32 })
}

/// Maps one Apple face to [`FaceGeom`].
///
/// Region points are normalized to the *face bbox* (bottom-left origin), per
/// `VNFaceLandmarkRegion2D` semantics: they are denormalized through the bbox
/// and flipped to top-left image coordinates.
fn map_face(face: &apple_vision::face_landmarks::FaceWithLandmarks) -> Option<FaceGeom> {
    use apple_vision::face_landmarks::LandmarkPoint as LP;
    let bb = &face.bounding_box;
    if !(bb.width > 0.0 && bb.height > 0.0) {
        return None;
    }
    // Region (u, v) in bbox space, bottom-left -> image top-left normalized.
    let img_pt = |p: &LP| Point { x: (bb.x + p.x * bb.width) as f32, y: (1.0 - (bb.y + p.y * bb.height)) as f32 };
    let region = |pts: &[LP]| pts.iter().map(img_pt).collect::<Vec<_>>();
    let req = |pts: &[LP]| -> Option<Vec<Point>> { if pts.is_empty() { None } else { Some(region(pts)) } };
    let left_eye = req(&face.left_eye)?;
    let right_eye = req(&face.right_eye)?;
    let nose = req(&face.nose)?;
    let lips = req(&face.outer_lips)?;
    let oval = if face.face_contour.is_empty() { Vec::new() } else { region(&face.face_contour) };
    let left_c = centroid(&left_eye)?;
    let right_c = centroid(&right_eye)?;
    let nose_c = centroid(&nose)?;
    let mouth_c = centroid(&lips)?;
    let eye_dist = left_c.dist(right_c).max(1e-4);
    // Eye radius from horizontal spread of the eye outline.
    let spread = |pts: &[Point]| {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in pts {
            lo = lo.min(p.x);
            hi = hi.max(p.x);
        }
        ((hi - lo) * 0.5).max(eye_dist * 0.1)
    };
    let eye_radius = spread(&left_eye).max(spread(&right_eye));
    // Mouth corners = extreme-x lip points.
    let mut mouth_left = lips[0];
    let mut mouth_right = lips[0];
    for p in &lips {
        if p.x < mouth_left.x {
            mouth_left = *p;
        }
        if p.x > mouth_right.x {
            mouth_right = *p;
        }
    }
    // Jaw = contour points below the mouth; chin = lowest contour point.
    let mut jaw: Vec<Point> = oval.iter().copied().filter(|p| p.y > mouth_c.y).collect();
    jaw.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(core::cmp::Ordering::Equal));
    let chin = oval.iter().copied().max_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(core::cmp::Ordering::Equal))?;
    Some(FaceGeom {
        bbox: flip_bbox(bb.x, bb.y, bb.width, bb.height),
        left_eye: left_c,
        right_eye: right_c,
        eye_radius,
        nose_tip: nose_c,
        mouth_left,
        mouth_right,
        mouth_center: mouth_c,
        jaw,
        chin,
        oval,
        has_dense: true,
    })
}

fn joint(joints: &std::collections::HashMap<String, apple_vision::body_pose::JointPoint>, name: &str, min_confidence: f32) -> Option<Point> {
    // Bridge quirk: keys arrive wrapped as "VNRecognizedPointKey(_rawValue: <name>)".
    joints.iter().find_map(|(key, j)| {
        let bare = key.strip_prefix("VNRecognizedPointKey(_rawValue: ").and_then(|s| s.strip_suffix(')')).unwrap_or(key.as_str());
        if bare != name || j.confidence < min_confidence {
            return None;
        }
        if !j.x.is_finite() || !j.y.is_finite() {
            return None;
        }
        // Bottom-left -> top-left.
        Some(Point { x: j.x as f32, y: (1.0 - j.y) as f32 })
    })
}

fn map_body(pose: &apple_vision::body_pose::DetectedBodyPose) -> BodyGeom {
    let joints = &pose.joints;
    let j = |name: &str| joint(joints, name, 0.25);
    // Raw Apple joint-name keys (see HumanBodyPoseJointName).
    BodyGeom {
        bbox: flip_bbox(pose.bounding_box.x, pose.bounding_box.y, pose.bounding_box.width, pose.bounding_box.height),
        joints: Joints {
            nose: j("head_joint"),
            left_shoulder: j("left_shoulder_1_joint"),
            right_shoulder: j("right_shoulder_1_joint"),
            left_hip: j("left_upLeg_joint"),
            right_hip: j("right_upLeg_joint"),
            left_knee: j("left_leg_joint"),
            right_knee: j("right_leg_joint"),
            left_ankle: j("left_foot_joint"),
            right_ankle: j("right_foot_joint"),
        },
    }
}

fn map_mask(mask: &apple_vision::segmentation::SegmentationMask) -> Option<PersonMask> {
    if mask.width == 0 || mask.height == 0 {
        return None;
    }
    if mask.bytes.len() < mask.height * mask.bytes_per_row {
        return None;
    }
    // Row 0 is the image top (verified empirically on real photos).
    let mut data = vec![0.0f32; mask.width * mask.height];
    for y in 0..mask.height {
        for x in 0..mask.width {
            data[y * mask.width + x] = mask.bytes[y * mask.bytes_per_row + x] as f32 / 255.0;
        }
    }
    Some(PersonMask { w: mask.width, h: mask.height, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_mapping_respects_confidence() {
        use apple_vision::body_pose::JointPoint as JP;
        let mut map = std::collections::HashMap::new();
        map.insert("left_foot_joint".to_string(), JP { x: 0.4, y: 0.2, confidence: 0.9 });
        map.insert("right_foot_joint".to_string(), JP { x: 0.6, y: 0.2, confidence: 0.1 });
        let good = joint(&map, "left_foot_joint", 0.25).unwrap();
        assert_eq!((good.x, good.y), (0.4, 0.8)); // y flipped
        assert!(joint(&map, "right_foot_joint", 0.25).is_none());
        assert!(joint(&map, "missing_joint", 0.25).is_none());
    }

    #[test]
    fn bbox_flip_is_correct() {
        // Bottom-left (0.2, 0.3, 0.4 x 0.5) -> top-left [0.2, 0.2, 0.6, 0.7].
        assert_eq!(flip_bbox(0.2, 0.3, 0.4, 0.5), [0.2, 0.2, 0.6, 0.7]);
    }

    #[test]
    fn png_roundtrip_dimensions() {
        let rgb = vec![128u8; 16 * 12 * 3];
        let png = encode_png_rgb8(&rgb, 16, 12).unwrap();
        assert!(png.len() > 8);
        assert_eq!(&png[1..4], b"PNG");
        assert!(encode_png_rgb8(&rgb, 0, 12).is_err());
        assert!(encode_png_rgb8(&[1u8, 2], 16, 12).is_err());
    }
}
