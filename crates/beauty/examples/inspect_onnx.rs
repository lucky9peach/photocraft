//! Inspect an ONNX file: inputs/outputs facts (+ optional single run).
//! Usage: cargo run -p photocraft-beauty --example inspect_onnx -- <model.onnx> [H W]
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: inspect_onnx <model.onnx> [H W]");
        std::process::exit(2);
    }
    let bytes = std::fs::read(PathBuf::from(&args[1])).expect("read model");
    println!("bytes: {}", bytes.len());
    let model = photocraft_beauty::detect::TractOnnx::load(&bytes).expect("load model");
    let (inputs, outputs) = model.facts();
    for (name, shape, dt) in &inputs {
        println!("in  {name} shape={shape:?} dtype={dt}");
    }
    for (name, shape, dt) in &outputs {
        println!("out {name} shape={shape:?} dtype={dt}");
    }
    if args.len() >= 4 {
        let h: usize = args[2].parse().expect("H");
        let w: usize = args[3].parse().expect("W");
        let input = vec![0.5f32; 3 * h * w];
        let t0 = std::time::Instant::now();
        let outs = model.run_chw(&input, 1, 3, h, w).expect("run");
        println!("run ok in {:?}; {} outputs", t0.elapsed(), outs.len());
        for (i, (data, dims)) in outs.iter().enumerate() {
            let finite = data.iter().filter(|v| v.is_finite()).count();
            println!("out{i}: dims={dims:?} len={} finite={finite}", data.len());
        }
    }
}
