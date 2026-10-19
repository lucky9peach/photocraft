# Portrait Beautify — PhotoCraft WebAssembly plug-in

Skin smoothing, blemish fade, brighten, saturation and sharpen, gated to skin tones, for
**Filter › Plug-ins** in PhotoCraft. Fully local and offline: the module has no imports, no
allocator, no host callbacks and no network — it only touches the pixel band it is handed.

This is the plug-in route (one `.wasm` file, installable from the app). It complements — it does
not replace — the native `photocraft-beauty` crate, which additionally does face/body detection and
geometric reshaping (eye enlarge, jaw/waist/leg warps). Those need landmarks and whole-image warps,
which plug-in ABI v1 cannot express.

## Build

```sh
rustup target add wasm32-unknown-unknown
./build.sh     # -> target/wasm32-unknown-unknown/release/photocraft_plugin_portrait_beautify.wasm
```

## Install (in the app)

**Filter › Plug-ins › Install Plug-in…** and pick the `.wasm` file. The plug-in is installed for
the running session; it then appears in the same submenu, with a generated dialog and live preview.

To load it automatically at every start, put the file in a folder and set
**Edit › Preferences › Plug-ins › Use Additional Plug-ins Folder** plus *Additional Plug-ins
Folder* to that folder: every `*.wasm` in it is installed at startup.

## Parameters

| Slider | What it does |
|---|---|
| `skinSmooth` (0–100) | Blends each pixel toward its neighbourhood mean where the pixel is flat (edge-preserving: edges and real texture survive). |
| `textureKeep` (0–100) | How much of the original fine detail is kept back (100 = none of the smoothing's detail loss). |
| `blemishFade` (0–100) | Finds pixels that differ from a local background and blends a small blur into them (spots a few pixels wide). |
| `brighten` (0–100) | Gamma lift of the skin (`x^(1−0.35s)`, approximated by a fitted polynomial). |
| `saturation` (−100–100) | Saturation change about the pixel's maximum, skin only. |
| `sharpen` (0–100) | Unsharp mask, skin only. |
| `skinOnly` (bool) | Off = apply to the whole image instead of skin-toned pixels. |

Defaults are a one-click "Natural" look (40 / 70 / 35 / 20 / 5 / 12 / on).

## What it cannot do (ABI limits, not laziness)

* **No face or body detection.** The gate is a per-pixel chroma rule (Chai & Ngan YCbCr ranges), so
  on photos with skin-toned backgrounds the effect applies there too. **Tip:** make a rough
  selection around the subject (or use Select Subject) before running the plug-in — the host blends
  the result with the original by the selection, so the effect stays inside it.
* **No geometric reshaping** (eye enlarge, jaw/waist/leg slim, leg lengthen): those need landmarks
  and warps of hundreds of pixels, which ABI v1 (stateless bands, no cross-band state) cannot do.
* **No eye/lip local contrast** (needs landmarks). Grayscale documents get the channel-agnostic
  passes (smoothing, blemish, sharpen); CMYK/Lab documents get those too, with no skin gate and no
  colour work.

## Performance

Bands run in parallel and every neighbourhood pass is a separable box blur with a sliding window
(constant cost per pixel whatever the radius). Measured through `plugin.run` (release host, Apple
silicon): 1 MP ≈ 2 s, 3.4 MP ≈ 3 s, 28 MP ≈ 24 s — the host's budget for one run is 60 s, so very
large photos (≳60 MP) are the practical limit; turning off stages that are at 0 is free.

## Layout

`src/lib.rs` is the whole plug-in: ABI exports (`pc_abi_version`, `pc_manifest`, `pc_alloc`,
`pc_filter`), a tiny JSON reader for the parameters (no `serde`, no allocator), stride-aware box
blurs, and the five passes. `no_std`, `panic = "abort"`, ~8.5 KB built.
