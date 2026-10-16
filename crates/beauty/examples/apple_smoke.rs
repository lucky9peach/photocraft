//! Apple Vision smoke test on a real portrait (scratch only, not committed data).
//! Usage: cargo run -p photocraft-beauty --example apple_smoke -- <photo.png>
#[cfg(target_vendor = "apple")]
use std::path::PathBuf;

#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("apple_smoke is macOS-only (Apple Vision backend)");
    std::process::exit(2);
}

#[cfg(target_vendor = "apple")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: apple_smoke <photo.png>");
        std::process::exit(2);
    }
    let img = image::open(PathBuf::from(&args[1])).expect("decode input");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    println!("photo: {w}x{h}");

    let t0 = std::time::Instant::now();
    let det = photocraft_beauty::apple::detect_rgb8(rgb.as_raw(), w, h).expect("detect");
    println!("detect total: {:?}", t0.elapsed());
    println!("faces: {}", det.faces.len());
    assert!(!det.faces.is_empty(), "expected >= 1 face");
    let f = &det.faces[0];
    println!(
        "face0 bbox={:.2?} eyes=({:.2},{:.2})/({:.2},{:.2}) nose=({:.2},{:.2}) mouth=({:.2},{:.2}) jaw_pts={} oval_pts={}",
        f.bbox,
        f.left_eye.x,
        f.left_eye.y,
        f.right_eye.x,
        f.right_eye.y,
        f.nose_tip.x,
        f.nose_tip.y,
        f.mouth_center.x,
        f.mouth_center.y,
        f.jaw.len(),
        f.oval.len()
    );
    // Geometric sanity in top-left normalized coords.
    assert!(f.left_eye.x < f.right_eye.x, "eye order");
    assert!(f.left_eye.y < f.nose_tip.y && f.nose_tip.y < f.mouth_center.y, "vertical order");
    assert!(f.chin.y > f.mouth_center.y, "chin below mouth");
    assert!(f.jaw.len() >= 4, "jaw polyline");
    assert!(f.oval.len() >= 8, "oval polygon");
    assert!(f.eye_radius > 0.005 && f.eye_radius < 0.2, "eye radius sane");

    println!("bodies: {}", det.bodies.len());
    for (i, b) in det.bodies.iter().enumerate() {
        let n = [
            &b.joints.nose,
            &b.joints.left_shoulder,
            &b.joints.right_shoulder,
            &b.joints.left_hip,
            &b.joints.right_hip,
            &b.joints.left_knee,
            &b.joints.right_knee,
            &b.joints.left_ankle,
            &b.joints.right_ankle,
        ]
        .into_iter()
        .filter(|j| j.is_some())
        .count();
        println!("  body{i}: {n}/9 key joints");
    }

    let mask = det.person_mask.as_ref().expect("person mask");
    let mean = mask.data.iter().sum::<f32>() / mask.data.len() as f32;
    println!("person mask: {}x{} mean={mean:.3}", mask.w, mask.h);
    assert!(mean > 0.01 && mean < 0.99, "mask must be non-trivial");

    // Orientation check: person confined to the TOP half => mass must be on top.
    let mut top = vec![0u8; (w as usize) * (h as usize) * 3];
    let half = h as usize / 2;
    for y in 0..half {
        let src_row = y * 2;
        if src_row >= h as usize {
            break;
        }
        for x in 0..w as usize {
            for c in 0..3 {
                top[(y * w as usize + x) * 3 + c] = rgb.as_raw()[(src_row * w as usize + x) * 3 + c];
            }
        }
    }
    let det2 = photocraft_beauty::apple::detect_rgb8(&top, w, h).expect("detect top-half");
    let m2 = det2.person_mask.as_ref().expect("mask2");
    let (mut mass_top, mut mass_bot) = (0.0f32, 0.0f32);
    for y in 0..m2.h {
        for x in 0..m2.w {
            let v = m2.data[y * m2.w + x];
            if y < m2.h / 2 {
                mass_top += v;
            } else {
                mass_bot += v;
            }
        }
    }
    println!("top-half composite: mass_top={mass_top:.0} mass_bot={mass_bot:.0}");
    assert!(mass_top > mass_bot * 1.5, "mask orientation flipped?");
    println!("APPLE_SMOKE OK");
}
