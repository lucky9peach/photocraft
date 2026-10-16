//! Dump raw Apple body-pose joints (scratch debugging).
#[cfg(target_vendor = "apple")]
use std::path::PathBuf;

#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("pose_dump is macOS-only (Apple Vision backend)");
    std::process::exit(2);
}

#[cfg(target_vendor = "apple")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img = image::open(PathBuf::from(&args[1])).expect("decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let path = std::env::temp_dir().join("pose_dump.png");
    img.save(&path).expect("save");
    let poses = apple_vision::body_pose::detect_human_body_pose_in_path(&path).expect("pose");
    let _ = std::fs::remove_file(&path);
    println!("img {w}x{h} poses: {}", poses.len());
    for (i, p) in poses.iter().enumerate() {
        println!("pose{i} bbox={:?} conf={}", p.bounding_box, p.confidence);
        let mut keys: Vec<_> = p.joints.iter().collect();
        keys.sort_by(|a, b| a.0.cmp(b.0));
        for (k, j) in keys {
            println!("  {k} x={:.3} y={:.3} c={:.2}", j.x, j.y, j.confidence);
        }
    }
}
