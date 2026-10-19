//! Portrait Beautify: a PhotoCraft filter plug-in (plug-in ABI v1, `docs/plugins.md`).
//!
//! Skin smoothing, blemish fade, brighten, saturation and sharpen, gated to skin tones by a
//! per-pixel chroma rule. Everything is local and offline: the module has no imports, no
//! allocator and no host callbacks, and it never reads anything outside the band it is handed.
//!
//! What it deliberately does **not** do (the sandbox cannot): face/body detection and geometric
//! reshaping (eye enlarge, jaw/waist/leg warps). Those need landmarks and whole-image warps, which
//! ABI v1 has no way to express; they live in the native `photocraft-beauty` crate instead.
//!
//! Performance notes (the host gives a filter run 60 s of wall clock for the whole image, and the
//! interpreter costs ~10–100× native per instruction):
//!
//! * Every neighbourhood pass is a separable box blur with a sliding window: constant cost per
//!   pixel whatever the radius, and only *two* passes over the interleaved buffer for all colour
//!   channels at once (no per-channel extract/merge passes).
//! * Hot loops use iterators/`chunks_exact` so the wasm stays free of bounds checks.
//! * Smoothing is a local-difference blend (`weight = 1 - d/(d + K)`, `d` = how far the pixel is
//!   from its neighbourhood mean): flat areas — skin — are smoothed, edges and texture survive.
//! * Brighten approximates the native gamma lift (`x^(1-0.35s)`) with a fitted polynomial, because
//!   `no_std` has no `powf`; the error is under ~2 % of full scale.
//! * Radii scale with the *canvas* size (from `_image.canvasWidth/canvasHeight`), never with the
//!   band size, so bands agree and the result has no seams.

#![no_std]

use core::arch::wasm32;
use core::sync::atomic::{AtomicU32, Ordering};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    wasm32::unreachable()
}

const MANIFEST: &str = r#"{
  "id": "dev.portraitlab.beautify",
  "name": "Portrait Beautify…",
  "version": "1.0.0",
  "kind": "filter",
  "author": "PortraitLab",
  "description": "Skin smoothing, blemish fade, brighten, saturation and sharpen, gated to skin tones. Fully offline, no face detection.",
  "params": {
    "skinSmooth": {"type": "number", "min": 0, "max": 100, "default": 40},
    "textureKeep": {"type": "number", "min": 0, "max": 100, "default": 70},
    "blemishFade": {"type": "number", "min": 0, "max": 100, "default": 35},
    "brighten": {"type": "number", "min": 0, "max": 100, "default": 20},
    "saturation": {"type": "number", "min": -100, "max": 100, "default": 5},
    "sharpen": {"type": "number", "min": 0, "max": 100, "default": 12},
    "skinOnly": {"type": "bool", "default": true}
  },
  "overlap": 16,
  "area": "content"
}"#;

/// Bit 8 of `format`: the last channel is straight alpha.
const HAS_ALPHA: u32 = 1 << 8;
/// Colour mode lives in bits 16–23, numbered as in the PSD file format.
const MODE_SHIFT: u32 = 16;
const MODE_GRAYSCALE: u32 = 1;
const MODE_RGB: u32 = 3;

/// Widest colour-channel count the running sums handle (RGBA and CMYK+alpha fit).
const MAX_CH: usize = 5;

/// Local-difference constant of the smoothing weight (≈13/255).
const SMOOTH_K: f32 = 0.05;
/// Blemish background scale and the radius of the smoothing applied to a spot.
const BLEMISH_BG: usize = 10;
/// Radius of the small blur shared by the blemish fade and the sharpen.
const SPOT_RADIUS: usize = 3;
/// Mask feather radius.
const MASK_FEATHER: usize = 2;
/// (The sharpen reuses [`SPOT_RADIUS`].)

/// Scratch region (one per instance; each band runs in a fresh instance).
static SCRATCH: AtomicU32 = AtomicU32::new(0);
static SCRATCH_SAMPLES: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------- ABI ----

#[unsafe(no_mangle)]
pub extern "C" fn pc_abi_version() -> u32 {
    1
}

/// `len << 32 | ptr` of the UTF-8 manifest JSON.
#[unsafe(no_mangle)]
pub extern "C" fn pc_manifest() -> u64 {
    ((MANIFEST.len() as u64) << 32) | MANIFEST.as_ptr() as u64
}

/// A block of `size` bytes: fresh pages from `memory.grow` (nothing is ever freed: each band gets
/// a fresh instance). Returns 0 when the host's memory cap is reached.
#[unsafe(no_mangle)]
pub extern "C" fn pc_alloc(size: u32) -> u32 {
    grow(size as usize)
}

fn grow(size: usize) -> u32 {
    let pages = size.div_ceil(65536);
    let old = wasm32::memory_grow(0, pages);
    if old == usize::MAX { 0 } else { (old * 65536) as u32 }
}

/// Scratch for `samples` `f32` samples, grown on first use and reused if a later call needs less.
fn scratch_for(samples: usize) -> u32 {
    let need = samples * 4;
    if SCRATCH_SAMPLES.load(Ordering::Relaxed) as usize >= samples {
        return SCRATCH.load(Ordering::Relaxed);
    }
    let base = grow(need);
    if base == 0 {
        return 0;
    }
    SCRATCH.store(base, Ordering::Relaxed);
    SCRATCH_SAMPLES.store(samples as u32, Ordering::Relaxed);
    base
}

/// A scratch slice of `len` samples at `offset` samples past `base`; regions never overlap.
///
/// # Safety
/// `base` must come from [`scratch_for`] for at least `offset + len` samples.
unsafe fn slice_at(base: u32, offset: usize, len: usize) -> &'static mut [f32] {
    let ptr = (base as usize + offset * 4) as *mut f32;
    // SAFETY: the caller guarantees the region.
    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

// ------------------------------------------------------- parameters ----

/// Index just past the `:` of `"key":` in `p`.
fn find(p: &[u8], key: &str) -> Option<usize> {
    let k = key.as_bytes();
    if k.is_empty() || p.len() < k.len() + 2 {
        return None;
    }
    let mut i = 0;
    while i + k.len() + 2 <= p.len() {
        if p[i] == b'"' && &p[i + 1..i + 1 + k.len()] == k && p[i + 1 + k.len()] == b'"' {
            let mut j = i + 2 + k.len();
            while j < p.len() && p[j] != b':' {
                j += 1;
            }
            return (j < p.len()).then_some(j + 1);
        }
        i += 1;
    }
    None
}

fn skip_ws(p: &[u8], mut i: usize) -> usize {
    while i < p.len() && matches!(p[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

fn parse_float(s: &[u8]) -> Option<f32> {
    let mut i = 0;
    let mut sign = 1.0f32;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        if s[i] == b'-' {
            sign = -1.0;
        }
        i += 1;
    }
    let mut v = 0.0f32;
    let mut any = false;
    while i < s.len() && s[i].is_ascii_digit() {
        v = v * 10.0 + f32::from(s[i] - b'0');
        i += 1;
        any = true;
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        let mut scale = 0.1f32;
        while i < s.len() && s[i].is_ascii_digit() {
            v += f32::from(s[i] - b'0') * scale;
            scale *= 0.1;
            i += 1;
            any = true;
        }
    }
    if any && i < s.len() && (s[i] == b'e' || s[i] == b'E') {
        i += 1;
        let mut esign = 1i32;
        if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
            if s[i] == b'-' {
                esign = -1;
            }
            i += 1;
        }
        let mut e = 0i32;
        while i < s.len() && s[i].is_ascii_digit() {
            e = (e * 10 + i32::from(s[i] - b'0')).min(38);
            i += 1;
        }
        for _ in 0..e {
            v = if esign > 0 { v * 10.0 } else { v / 10.0 };
        }
    }
    any.then_some(sign * v)
}

fn num(p: &[u8], key: &str, default: f32) -> f32 {
    match find(p, key) {
        Some(at) => {
            let start = skip_ws(p, at);
            let mut end = start;
            while end < p.len() && (p[end].is_ascii_digit() || matches!(p[end], b'-' | b'+' | b'.' | b'e' | b'E')) {
                end += 1;
            }
            parse_float(&p[start..end]).unwrap_or(default)
        }
        None => default,
    }
}

fn flag(p: &[u8], key: &str, default: bool) -> bool {
    match find(p, key) {
        Some(at) => {
            let i = skip_ws(p, at);
            p[i..].starts_with(b"true")
        }
        None => default,
    }
}

// ------------------------------------------------------------ blur ----

/// Horizontal box blur of `ch` interleaved colour channels (stride `n`), clamped edges.
fn box_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, n: usize, ch: usize, r: usize) {
    if r == 0 {
        for (s, d) in src.chunks_exact(n).zip(dst.chunks_exact_mut(n)) {
            d[..ch].copy_from_slice(&s[..ch]);
        }
        return;
    }
    for (rs, rd) in src.chunks_exact(w * n).zip(dst.chunks_exact_mut(w * n)).take(h) {
        let mut sum = [0.0f32; MAX_CH];
        for x in 0..=r.min(w - 1) {
            let o = x * n;
            for (acc, s) in sum.iter_mut().zip(&rs[o..o + ch]) {
                *acc += *s;
            }
        }
        for x in 0..w {
            let o = x * n;
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let inv = 1.0 / (x1 - x0 + 1) as f32;
            for (d, s) in rd[o..o + ch].iter_mut().zip(&sum[..ch]) {
                *d = *s * inv;
            }
            if x + r + 1 <= w - 1 {
                let a = (x + r + 1) * n;
                for (acc, s) in sum.iter_mut().zip(&rs[a..a + ch]) {
                    *acc += *s;
                }
            }
            if x >= r {
                let b = (x - r) * n;
                for (acc, s) in sum.iter_mut().zip(&rs[b..b + ch]) {
                    *acc -= *s;
                }
            }
        }
    }
}

/// Vertical box blur of `ch` interleaved colour channels (stride `n`), clamped edges.
fn box_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, n: usize, ch: usize, r: usize) {
    if r == 0 {
        for (s, d) in src.chunks_exact(n).zip(dst.chunks_exact_mut(n)) {
            d[..ch].copy_from_slice(&s[..ch]);
        }
        return;
    }
    let stride = w * n;
    for x in 0..w {
        let col = x * n;
        let mut sum = [0.0f32; MAX_CH];
        for y in 0..=r.min(h - 1) {
            let o = y * stride + col;
            for (acc, s) in sum.iter_mut().zip(&src[o..o + ch]) {
                *acc += *s;
            }
        }
        for y in 0..h {
            let o = y * stride + col;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let inv = 1.0 / (y1 - y0 + 1) as f32;
            for (d, s) in dst[o..o + ch].iter_mut().zip(&sum[..ch]) {
                *d = *s * inv;
            }
            if y + r + 1 <= h - 1 {
                let a = (y + r + 1) * stride + col;
                for (acc, s) in sum.iter_mut().zip(&src[a..a + ch]) {
                    *acc += *s;
                }
            }
            if y >= r {
                let b = (y - r) * stride + col;
                for (acc, s) in sum.iter_mut().zip(&src[b..b + ch]) {
                    *acc -= *s;
                }
            }
        }
    }
}

/// Separable box blur: `src` → `dst`, using `tmp` as the intermediate buffer.
fn blur(src: &[f32], tmp: &mut [f32], dst: &mut [f32], w: usize, h: usize, n: usize, ch: usize, r: usize) {
    box_h(src, tmp, w, h, n, ch, r);
    box_v(tmp, dst, w, h, n, ch, r);
}

// ------------------------------------------------------------ tone ----

/// Brighten curve: the native gamma lift `x^(1 - 0.35s)` fitted as `x + s * x(1-x)(A + Bx + Cx²)`
/// (max error ~2 % of full scale in deep shadows; `no_std` has no `powf`).
fn brighten_curve(x: f32, s: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let lift = x * (1.0 - x) * (1.538 - 2.835 * x + 1.709 * x * x);
    (x + s * lift).clamp(0.0, 1.0)
}


/// Per-pixel skin rule (Chai & Ngan YCbCr ranges) into `mask`; 1 = skin.
fn skin_mask(px: &[f32], mask: &mut [f32], n: usize) {
    for (m, p) in mask.iter_mut().zip(px.chunks_exact(n)) {
        let (r, g, b) = (p[0] * 255.0, p[1] * 255.0, p[2] * 255.0);
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        *m = if cb > 77.0 && cb < 127.0 && cr > 133.0 && cr < 173.0 { 1.0 } else { 0.0 };
    }
}

// -------------------------------------------------------- pipeline ----

/// Filters `width × height` pixels of `channels` interleaved `f32` samples at `buf`.
///
/// # Safety
/// The host guarantees `buf..buf + len` is a block it got from `pc_alloc` holding `len / 4`
/// samples, and that `params..params + params_len` is readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pc_filter(
    buf: *mut f32,
    len: u32,
    width: u32,
    height: u32,
    channels: u32,
    format: u32,
    params: *const u8,
    params_len: u32,
) -> i32 {
    let n = channels as usize;
    let (w, h) = (width as usize, height as usize);
    if n == 0 || w == 0 || h == 0 || n > MAX_CH + 1 {
        return 0;
    }
    let Some(pixels) = w.checked_mul(h) else { return 1 };
    let samples = len as usize / 4;
    if buf.is_null() || samples < pixels * n {
        return 1;
    }
    // SAFETY: the host's contract (see the doc comment).
    let px = unsafe { core::slice::from_raw_parts_mut(buf, pixels * n) };
    let pj: &[u8] = if params.is_null() || params_len == 0 {
        &[]
    } else {
        // SAFETY: the host's contract.
        unsafe { core::slice::from_raw_parts(params, params_len as usize) }
    };

    let smooth = num(pj, "skinSmooth", 40.0).clamp(0.0, 100.0) / 100.0;
    let texture = num(pj, "textureKeep", 70.0).clamp(0.0, 100.0) / 100.0;
    let blemish = num(pj, "blemishFade", 35.0).clamp(0.0, 100.0) / 100.0;
    let brighten = num(pj, "brighten", 20.0).clamp(0.0, 100.0) / 100.0;
    let saturation = num(pj, "saturation", 5.0).clamp(-100.0, 100.0) / 100.0;
    let sharpen = num(pj, "sharpen", 12.0).clamp(0.0, 100.0) / 100.0;
    let skin_only = flag(pj, "skinOnly", true);
    if smooth <= 0.0 && blemish <= 0.0 && brighten <= 0.0 && saturation == 0.0 && sharpen <= 0.0 {
        return 0;
    }

    let alpha = format & HAS_ALPHA != 0;
    let mode = (format >> MODE_SHIFT) & 0xFF;
    let colour_ch = if alpha { n - 1 } else { n };
    // Skin detection and saturation need an RGB colour model (samples are in the document's own
    // model, with no conversion). Grayscale and the other modes get the channel-agnostic passes.
    let rgb = mode == MODE_RGB && colour_ch >= 3;
    let gray = mode == MODE_GRAYSCALE;
    let do_skin = skin_only && rgb;
    let do_tone = rgb;
    let do_blemish = blemish > 0.0 && (rgb || gray);

    // Radii follow the canvas, never the band, so every band agrees.
    let canvas_w = num(pj, "canvasWidth", w as f32).max(1.0);
    let canvas_h = num(pj, "canvasHeight", h as f32).max(1.0);
    let r_smooth = ((canvas_w.min(canvas_h) / 220.0) as usize).clamp(2, 8);

    // Scratch: two interleaved buffers (blur result + temp) and five single-channel planes.
    let interleaved = pixels * n;
    let (off_t, off_m, off_m2, off_v, off_mask, off_aux) =
        (interleaved, 2 * interleaved, 2 * interleaved + pixels, 2 * interleaved + 2 * pixels, 2 * interleaved + 3 * pixels, 2 * interleaved + 4 * pixels);
    let base = scratch_for(2 * interleaved + 5 * pixels);
    if base == 0 {
        return 2;
    }
    // SAFETY: `base` covers every offset below; the regions never overlap.
    let (p, t, m, m2, v, mask, aux) = unsafe {
        (
            slice_at(base, 0, interleaved),
            slice_at(base, off_t, interleaved),
            slice_at(base, off_m, pixels),
            slice_at(base, off_m2, pixels),
            slice_at(base, off_v, pixels),
            slice_at(base, off_mask, pixels),
            slice_at(base, off_aux, pixels),
        )
    };

    // ---- skin mask (feathered, so every effect fades out at the skin boundary) ----
    if do_skin {
        skin_mask(px, mask, n);
        blur(mask, m, v, w, h, 1, 1, MASK_FEATHER);
        mask.copy_from_slice(&v);
    } else {
        for slot in mask.iter_mut() {
            *slot = 1.0;
        }
    }

    // ---- 1. skin smoothing: blend toward the neighbourhood mean where the pixel is flat ----
    if smooth > 0.0 {
        blur(px, t, p, w, h, n, colour_ch, r_smooth);
        let detail = 1.0 - texture * 0.5;
        for (i, pxp) in px.chunks_exact_mut(n).enumerate() {
            if alpha && pxp[n - 1] <= 0.0 {
                continue;
            }
            let bp = &p[i * n..i * n + colour_ch];
            let mut d = 0.0f32;
            for (o, b) in pxp[..colour_ch].iter().zip(bp) {
                d = d.max((o - b).abs());
            }
            let g = (mask[i] * smooth * (1.0 - d / (d + SMOOTH_K)) * detail).clamp(0.0, 1.0);
            if g <= 0.0 {
                continue;
            }
            for (o, b) in pxp[..colour_ch].iter_mut().zip(bp) {
                *o += (*b - *o) * g;
            }
        }
    }

    // ---- 2. one shared small blur for the blemish fade and the sharpen ----
    let small_blur = do_blemish || sharpen > 0.0;
    if small_blur {
        blur(px, t, p, w, h, n, colour_ch, SPOT_RADIUS);
    }

    // ---- 3. blemish fade: background threshold, feathered, blend the small blur ----
    if do_blemish {
        let lc = if rgb { 1 } else { 0 };
        for (lv, pxp) in m.iter_mut().zip(px.chunks_exact(n)) {
            *lv = if rgb { 0.299 * pxp[0] + 0.587 * pxp[1] + 0.114 * pxp[2] } else { pxp[lc] };
        }
        blur(m, aux, m2, w, h, 1, 1, BLEMISH_BG);
        let thresh = (30.0 - 18.0 * blemish) / 255.0;
        for (spot, (l, bg)) in aux.iter_mut().zip(m.iter().zip(m2.iter())) {
            *spot = if (l - bg).abs() > thresh { 1.0 } else { 0.0 };
        }
        // 3×3 feather of the spot mask (softer and much cheaper than a dilate).
        blur(aux, v, m2, w, h, 1, 1, 1);
        let blend = 0.4 + 0.6 * blemish;
        for i in 0..pixels {
            aux[i] = (m2[i] * mask[i] * blend).clamp(0.0, 1.0);
        }
        for (i, pxp) in px.chunks_exact_mut(n).enumerate() {
            let g = aux[i];
            if g <= 0.0 || (alpha && pxp[n - 1] <= 0.0) {
                continue;
            }
            let bp = &p[i * n..i * n + colour_ch];
            for (o, b) in pxp[..colour_ch].iter_mut().zip(bp) {
                *o += (*b - *o) * g;
            }
        }
    }

    // ---- 4. brighten + saturation in one pass (skin-gated) ----
    if do_tone && (brighten > 0.0 || saturation != 0.0) {
        let b_blend = 0.5 + 0.5 * brighten;
        let s_gate = (saturation.abs() * 2.0).min(1.0);
        for (i, pxp) in px.chunks_exact_mut(n).enumerate() {
            if alpha && pxp[n - 1] <= 0.0 {
                continue;
            }
            let m = mask[i];
            if m <= 0.0 {
                continue;
            }
            if brighten > 0.0 {
                let g = m * b_blend;
                for o in pxp[..3].iter_mut() {
                    *o += (brighten_curve(*o, brighten) - *o) * g;
                }
            }
            if saturation == 0.0 {
                continue;
            }
            let g = m * s_gate;
            let (r, gg, b) = (pxp[0], pxp[1], pxp[2]);
            let mx = r.max(gg).max(b);
            let mn = r.min(gg).min(b);
            if mx - mn <= 1e-6 || mx <= 1e-6 {
                continue;
            }
            let s = (mx - mn) / mx;
            let ns = if saturation > 0.0 { s * (1.0 + 0.6 * saturation) } else { s * (1.0 + 0.8 * saturation) };
            let k = (ns.clamp(0.0, 1.0) / s).clamp(0.0, 4.0);
            pxp[0] = r + ((mx + (r - mx) * k) - r) * g;
            pxp[1] = gg + ((mx + (gg - mx) * k) - gg) * g;
            pxp[2] = b + ((mx + (b - mx) * k) - b) * g;
        }
    }

    // ---- 5. sharpen (unsharp mask, skin-gated, reusing the small blur) ----
    if sharpen > 0.0 {
        let amount = 0.8 * sharpen;
        let blend = 0.5 + 0.5 * sharpen;
        for (i, pxp) in px.chunks_exact_mut(n).enumerate() {
            if alpha && pxp[n - 1] <= 0.0 {
                continue;
            }
            let g = (mask[i] * blend).clamp(0.0, 1.0);
            if g <= 0.0 {
                continue;
            }
            let bp = &p[i * n..i * n + colour_ch];
            for (o, b) in pxp[..colour_ch].iter_mut().zip(bp) {
                *o += amount * g * (*o - *b);
            }
        }
    }

    0
}
