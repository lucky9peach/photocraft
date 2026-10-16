# photocraft-beauty

Native (non-WASM) portrait/body beautification for PhotoCraft: parameters,
presets, feathered mask math, skin/light algorithms, a unified warp engine,
face/body geometry, and detection backends. The engine front-end lives in
`crates/engine/src/beauty_cmds.rs` (Filter › Beautify); this crate stays
pure (no I/O except the Apple Vision temp-PNG bridge) and independently
testable.

Modules:

- `params` — `BeautifyParams` (20 sliders, JSON round-trip) + presets
  (Natural / Cosplay / Editorial).
- `mask` — feather, union, subtract, masked blend, convex-hull fill,
  bilinear resample.
- `warp` — displacement `Field` (radial / pull / squeeze / stretch) +
  mask-gated bilinear remap; untouched pixels stay bit-identical.
- `skin` — guided smoothing, Lab tone uniform, dual-scale blemish fade.
- `light` — brighten, saturation, CLAHE eye detail, sharpen.
- `geom` — `FaceGeom`/`BodyGeom`, `face_warps`/`body_warps`, mask-driven
  torso/leg estimation (`complete_body_joints`).
- `apple` (macOS only) — Apple Vision backend, zero weights.
- `detect` (feature `onnx`, default on) — tract ONNX runtime + model slots.
  The engine builds with `default-features = false` (no tract) so wasm
  builds stay lean; enable `onnx` for the Linux/Windows backends.

Conventions: no panics on input (all fallible fns return `Error`), no
`unsafe`, no bit-depth assumptions (masks are always `f32` 0..1),
wasm32-compatible without the `onnx` feature.

See `docs/beauty-requirements.md` (roadmap + decisions) and
`docs/beauty-install.md` (build & try it yourself).
