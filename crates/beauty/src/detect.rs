//! Local ONNX inference (pure-Rust `tract`) + detector decoders.
//!
//! Models load from bytes (the engine owns files/downloads); nothing here
//! touches the filesystem, so the module stays wasm32-compatible.
//! Weight provenance + licenses live in `docs/MODELS_AND_LICENSES.md`-style
//! registry entries, see [`ModelSlot`].

use std::io::Cursor;

use tract_onnx::prelude::*;

use crate::Error;

/// A model slot: what the pipeline needs, and where verified weights come from.
#[derive(Clone, Debug)]
pub struct ModelSlot {
    pub key: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub download_url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
}

/// Face detection weights (YuNet 2023mar, MIT, hash-verified).
pub const YUNET_2023MAR: ModelSlot = ModelSlot {
    key: "yunet_2023mar",
    description: "YuNet face detector: box + 5 landmarks (eyes, nose, mouth corners).",
    license: "MIT (Shiqi Yu / libfacedetection, via opencv_zoo)",
    license_url: "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE",
    download_url: "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size_bytes: 232_589,
};

/// Single-person pose weights (MoveNet Lightning, Apache-2.0 card).
/// Converted-by-third-party copy; hash pins the exact bytes validated here.
pub const MOVENET_LIGHTNING: ModelSlot = ModelSlot {
    key: "movenet_lightning",
    description: "MoveNet single-pose lightning: 17 COCO joints.",
    license: "Apache-2.0 (per HuggingFace model card; Google MoveNet upstream)",
    license_url: "https://huggingface.co/Xenova/movenet-singlepose-lightning",
    download_url: "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
    sha256: "1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad",
    size_bytes: 9_413_268,
};

fn tract_err(what: &'static str, e: impl core::fmt::Display) -> Error {
    Error::InvalidParam(format!("{what}: {e}"))
}

/// An ONNX model runnable on CPU via tract.
type TractRunnable = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Named tensor facts: (name, dims, datum-type).
pub type TensorFacts = Vec<(String, Vec<i64>, String)>;

/// Model outputs: flat f32 samples + dims each.
pub type ModelOutputs = Vec<(Vec<f32>, Vec<usize>)>;

pub struct TractOnnx {
    model: TractRunnable,
}

impl TractOnnx {
    /// Loads (and optimizes) a model from ONNX bytes.
    pub fn load(bytes: &[u8]) -> Result<Self, Error> {
        let mut cursor = Cursor::new(bytes);
        let model = tract_onnx::onnx().model_for_read(&mut cursor).map_err(|e| tract_err("onnx parse", e))?;
        let model = model.into_optimized().map_err(|e| tract_err("onnx optimize", e))?;
        let runnable = model.into_runnable().map_err(|e| tract_err("onnx runnable", e))?;
        Ok(Self { model: runnable })
    }

    /// (inputs, outputs) as (name, shape, datum-type) for inspection.
    pub fn facts(&self) -> (TensorFacts, TensorFacts) {
        let model = self.model.model();
        let shape_of = |id: OutletId| -> (Vec<i64>, String) {
            model
                .outlet_fact(id)
                .map(|f| (f.shape.dims().iter().map(|d| d.to_i64().unwrap_or(-1)).collect(), format!("{:?}", f.datum_type)))
                .unwrap_or_else(|_| (Vec::new(), "?".to_string()))
        };
        let inputs: Vec<(String, Vec<i64>, String)> = model
            .input_outlets()
            .map(|outs| {
                outs.iter()
                    .enumerate()
                    .map(|(i, id)| {
                        let (shape, dt) = shape_of(*id);
                        (format!("in{i}"), shape, dt)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let outputs: Vec<(String, Vec<i64>, String)> = model
            .output_outlets()
            .map(|outs| {
                outs.iter()
                    .map(|id| {
                        let (shape, dt) = shape_of(*id);
                        let node = &model.nodes[id.node];
                        (format!("{}:{}", node.name, node.op.name()), shape, dt)
                    })
                    .collect()
            })
            .unwrap_or_default();
        (inputs, outputs)
    }

    /// Runs the model on one NCHW f32 tensor; returns each output as flat f32 + dims.
    pub fn run_chw(&self, input: &[f32], n: usize, c: usize, h: usize, w: usize) -> Result<ModelOutputs, Error> {
        if input.len() != n * c * h * w {
            return Err(Error::ShapeMismatch { what: "tract input", expected: n * c * h * w, found: input.len() });
        }
        let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((n, c, h, w), input.to_vec()).map_err(|e| tract_err("input tensor", e))?.into();
        let outputs = self.model.run(tvec!(tensor.into())).map_err(|e| tract_err("infer", e))?;
        let mut out = Vec::with_capacity(outputs.len());
        for value in outputs.iter() {
            let view = value.to_array_view::<f32>().map_err(|e| tract_err("output cast", e))?;
            let dims = view.shape().to_vec();
            out.push((view.iter().copied().collect(), dims));
        }
        Ok(out)
    }
}

/// IoU of two [x0,y0,x1,y1] boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { (inter / union).clamp(0.0, 1.0) }
}

/// Greedy NMS over `(score, index)` pairs with boxes; returns kept indices.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], threshold: f32, top_k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(core::cmp::Ordering::Equal));
    let mut kept = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        if kept.len() >= top_k.max(1) {
            break;
        }
        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }
            if iou(&boxes[i], &boxes[j]) > threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_keeps_best_and_drops_overlap() {
        let boxes = vec![[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]];
        let scores = vec![0.9, 0.8, 0.7];
        assert_eq!(nms(&boxes, &scores, 0.3, 100), vec![0, 2]);
        assert_eq!(nms(&boxes, &scores, 0.3, 1), vec![0]);
        assert!(nms(&[], &[], 0.3, 10).is_empty());
    }

    #[test]
    fn registry_slots_sane() {
        for slot in [YUNET_2023MAR, MOVENET_LIGHTNING] {
            assert_eq!(slot.sha256.len(), 64);
            assert!(slot.size_bytes > 1000);
            assert!(slot.download_url.starts_with("http"));
        }
    }
}
