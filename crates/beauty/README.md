# photocraft-beauty

Native (non-WASM) portrait/body beautification for PhotoCraft: parameters,
presets and feathered mask math shared by the beauty filters.

Stage 1 (this crate today): serializable `BeautifyParams`, the three built-in
presets (Natural / Cosplay / Editorial), and pure `f32` mask operators
(feather, union, subtract, masked blend, convex-hull rasterization) that work
on any buffer layout. Detection (face/body landmarks), skin smoothing and
geometric warps build on top of this in later stages — see
`docs/beauty-requirements.md` for the roadmap.

Conventions: no panics on input (all fallible fns return `Error`), no `unsafe`,
no bit-depth assumptions (masks are always `f32` 0..1), wasm32-compatible
(no threads, no FS, no system deps beyond `serde`).
