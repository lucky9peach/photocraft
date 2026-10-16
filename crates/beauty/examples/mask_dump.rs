//! Temporary debug: dump person mask + leg estimate for a photo.
//! Usage: cargo run -p photocraft-beauty --example mask_dump -- <photo.png> <out_mask.png>
#[cfg(target_vendor = "apple")]
use std::path::PathBuf;

#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("mask_dump is macOS-only");
    std::process::exit(2);
}

#[cfg(target_vendor = "apple")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img = image::open(PathBuf::from(&args[1])).expect("decode input");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let det = photocraft_beauty::apple::detect_rgb8(rgb.as_raw(), w, h).expect("detect");
    println!("faces={} bodies={}", det.faces.len(), det.bodies.len());
    for (i, b) in det.bodies.iter().enumerate() {
        println!("body{i} bbox={:?}", b.bbox);
        let j = &b.joints;
        for (name, p) in [
            ("nose", &j.nose),
            ("l_sho", &j.left_shoulder),
            ("r_sho", &j.right_shoulder),
            ("l_hip", &j.left_hip),
            ("r_hip", &j.right_hip),
            ("l_knee", &j.left_knee),
            ("r_knee", &j.right_knee),
            ("l_ank", &j.left_ankle),
            ("r_ank", &j.right_ankle),
        ] {
            println!("  {name}: {p:?}");
        }
    }
    let m = det.person_mask.as_ref().expect("mask");
    println!("mask {}x{} mean={:.4}", m.w, m.h, m.data.iter().sum::<f32>() / m.data.len() as f32);
    // Save mask as PNG for inspection.
    let bytes: Vec<u8> = m.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect();
    image::save_buffer(PathBuf::from(&args[2]), &bytes, m.w as u32, m.h as u32, image::ColorType::L8).expect("save mask");
    // Leg estimate on the mask's native grid.
    let legs = photocraft_beauty::geom::estimate_legs_from_mask(&m.data, m.w, m.h).expect("estimate");
    println!("legs: {legs:?}");
    // Row-mass profile (every 32 rows) to see the split.
    for y in (0..m.h).step_by(32) {
        let mut runs = 0;
        let mut in_run = false;
        for x in 0..m.w {
            let on = m.data[y * m.w + x] > 0.5;
            if on && !in_run {
                runs += 1;
            }
            in_run = on;
        }
        let mass: f32 = m.data[y * m.w..(y + 1) * m.w].iter().sum();
        println!("row {y}: runs={runs} mass={mass:.0}");
    }
}
