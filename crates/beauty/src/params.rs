//! Single source of truth for beautification parameters.
//!
//! Strengths are 0..100 (0 = effect disabled) except `saturation`, which is
//! -100..100 (0 = neutral). Geometry defaults to 0 (conservative, off).

use serde::{Deserialize, Serialize};

use crate::Error;

/// All beautification strengths. Every effect is independently tunable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeautifyParams {
    // --- skin ---
    pub skin_smooth: f32,
    pub skin_tone_uniform: f32,
    pub blemish_fade: f32,
    /// 0..100; higher keeps more skin texture.
    pub texture_keep: f32,
    // --- geometry (all default OFF) ---
    pub eye_enlarge: f32,
    pub jaw_slim: f32,
    pub nose_refine: f32,
    /// N1 melon-seed / V-line shaping.
    pub vline: f32,
    /// N3 mouth-corner lift.
    pub mouth_lift: f32,
    /// N5 apple-muscle lift.
    pub cheek_lift: f32,
    /// B1 torso slimming.
    pub slim_body: f32,
    /// B1 waist narrowing.
    pub waist_slim: f32,
    /// B2 leg slimming.
    pub slim_legs: f32,
    /// B3 leg lengthening.
    pub leg_lengthen: f32,
    /// B5 head/body proportion (subtle head shrink).
    pub head_ratio: f32,
    // --- light & detail ---
    pub brighten: f32,
    /// -100..100.
    pub saturation: f32,
    pub eye_detail: f32,
    pub sharpen: f32,
    /// N5 nasolabial-fold fade (localized smoothing).
    pub nasolabial_fade: f32,
}

impl Default for BeautifyParams {
    fn default() -> Self {
        Self {
            skin_smooth: 0.0,
            skin_tone_uniform: 0.0,
            blemish_fade: 0.0,
            texture_keep: 70.0,
            eye_enlarge: 0.0,
            jaw_slim: 0.0,
            nose_refine: 0.0,
            vline: 0.0,
            mouth_lift: 0.0,
            cheek_lift: 0.0,
            slim_body: 0.0,
            waist_slim: 0.0,
            slim_legs: 0.0,
            leg_lengthen: 0.0,
            head_ratio: 0.0,
            brighten: 0.0,
            saturation: 0.0,
            eye_detail: 0.0,
            sharpen: 0.0,
            nasolabial_fade: 0.0,
        }
    }
}
fn clamp_0_100(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(0.0, 100.0)
}

impl BeautifyParams {
    /// Returns a copy with every field clamped to its valid range.
    /// Non-finite values become 0 (effect disabled), never NaN.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut p = self.clone();
        p.skin_smooth = clamp_0_100(p.skin_smooth);
        p.skin_tone_uniform = clamp_0_100(p.skin_tone_uniform);
        p.blemish_fade = clamp_0_100(p.blemish_fade);
        p.texture_keep = clamp_0_100(p.texture_keep);
        p.eye_enlarge = clamp_0_100(p.eye_enlarge);
        p.jaw_slim = clamp_0_100(p.jaw_slim);
        p.nose_refine = clamp_0_100(p.nose_refine);
        p.vline = clamp_0_100(p.vline);
        p.mouth_lift = clamp_0_100(p.mouth_lift);
        p.cheek_lift = clamp_0_100(p.cheek_lift);
        p.slim_body = clamp_0_100(p.slim_body);
        p.waist_slim = clamp_0_100(p.waist_slim);
        p.slim_legs = clamp_0_100(p.slim_legs);
        p.leg_lengthen = clamp_0_100(p.leg_lengthen);
        p.head_ratio = clamp_0_100(p.head_ratio);
        p.brighten = clamp_0_100(p.brighten);
        p.saturation = if p.saturation.is_finite() { p.saturation.clamp(-100.0, 100.0) } else { 0.0 };
        p.eye_detail = clamp_0_100(p.eye_detail);
        p.sharpen = clamp_0_100(p.sharpen);
        p.nasolabial_fade = clamp_0_100(p.nasolabial_fade);
        p
    }

    /// Strict validation: rejects NaN/infinite and out-of-range values.
    pub fn validate(&self) -> Result<(), Error> {
        let pairs = [
            ("skin_smooth", self.skin_smooth, 0.0, 100.0),
            ("skin_tone_uniform", self.skin_tone_uniform, 0.0, 100.0),
            ("blemish_fade", self.blemish_fade, 0.0, 100.0),
            ("texture_keep", self.texture_keep, 0.0, 100.0),
            ("eye_enlarge", self.eye_enlarge, 0.0, 100.0),
            ("jaw_slim", self.jaw_slim, 0.0, 100.0),
            ("nose_refine", self.nose_refine, 0.0, 100.0),
            ("vline", self.vline, 0.0, 100.0),
            ("mouth_lift", self.mouth_lift, 0.0, 100.0),
            ("cheek_lift", self.cheek_lift, 0.0, 100.0),
            ("slim_body", self.slim_body, 0.0, 100.0),
            ("waist_slim", self.waist_slim, 0.0, 100.0),
            ("slim_legs", self.slim_legs, 0.0, 100.0),
            ("leg_lengthen", self.leg_lengthen, 0.0, 100.0),
            ("head_ratio", self.head_ratio, 0.0, 100.0),
            ("brighten", self.brighten, 0.0, 100.0),
            ("saturation", self.saturation, -100.0, 100.0),
            ("eye_detail", self.eye_detail, 0.0, 100.0),
            ("sharpen", self.sharpen, 0.0, 100.0),
            ("nasolabial_fade", self.nasolabial_fade, 0.0, 100.0),
        ];
        for (name, v, lo, hi) in pairs {
            if !v.is_finite() {
                return Err(Error::InvalidParam(format!("{name} is not finite")));
            }
            if v < lo || v > hi {
                return Err(Error::InvalidParam(format!("{name}={v} out of range {lo}..{hi}")));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn has_geometry(&self) -> bool {
        self.eye_enlarge > 0.0
            || self.jaw_slim > 0.0
            || self.nose_refine > 0.0
            || self.vline > 0.0
            || self.mouth_lift > 0.0
            || self.cheek_lift > 0.0
            || self.slim_body > 0.0
            || self.waist_slim > 0.0
            || self.slim_legs > 0.0
            || self.leg_lengthen > 0.0
            || self.head_ratio > 0.0
    }

    #[must_use]
    pub fn has_skin_work(&self) -> bool {
        self.skin_smooth > 0.0 || self.skin_tone_uniform > 0.0 || self.blemish_fade > 0.0 || self.nasolabial_fade > 0.0
    }

    #[must_use]
    pub fn has_light_work(&self) -> bool {
        self.brighten > 0.0 || self.saturation != 0.0 || self.eye_detail > 0.0 || self.sharpen > 0.0
    }

    /// True when processing would be an exact no-op.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        !(self.has_geometry() || self.has_skin_work() || self.has_light_work())
    }
}

/// Built-in preset names in UI order.
pub const BUILTIN_ORDER: [&str; 3] = ["Natural", "Cosplay", "Editorial"];

fn natural() -> BeautifyParams {
    BeautifyParams {
        skin_smooth: 30.0,
        skin_tone_uniform: 20.0,
        blemish_fade: 20.0,
        texture_keep: 80.0,
        brighten: 15.0,
        saturation: 5.0,
        eye_detail: 15.0,
        sharpen: 12.0,
        ..BeautifyParams::default()
    }
}

fn cosplay() -> BeautifyParams {
    BeautifyParams {
        skin_smooth: 55.0,
        skin_tone_uniform: 30.0,
        blemish_fade: 50.0,
        texture_keep: 60.0,
        eye_enlarge: 18.0,
        jaw_slim: 12.0,
        nose_refine: 10.0,
        vline: 8.0,
        mouth_lift: 10.0,
        cheek_lift: 0.0,
        slim_body: 0.0,
        waist_slim: 0.0,
        slim_legs: 0.0,
        leg_lengthen: 0.0,
        head_ratio: 0.0,
        brighten: 20.0,
        saturation: 12.0,
        eye_detail: 40.0,
        sharpen: 15.0,
        nasolabial_fade: 25.0,
    }
}

fn editorial() -> BeautifyParams {
    BeautifyParams {
        skin_smooth: 45.0,
        skin_tone_uniform: 40.0,
        blemish_fade: 35.0,
        texture_keep: 75.0,
        brighten: 30.0,
        saturation: 8.0,
        eye_detail: 30.0,
        sharpen: 25.0,
        ..BeautifyParams::default()
    }
}

/// Built-in presets: pure parameter sets, no algorithm logic.
pub type PresetEntry = (&'static str, fn() -> BeautifyParams);
pub const PRESETS: [PresetEntry; 3] = [("Natural", natural), ("Cosplay", cosplay), ("Editorial", editorial)];

/// Looks up a built-in preset by name.
pub fn builtin(name: &str) -> Result<BeautifyParams, Error> {
    for (preset_name, build) in PRESETS {
        if preset_name == name {
            return Ok(build());
        }
    }
    Err(Error::InvalidParam(format!("unknown preset: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_and_identity() {
        let p = BeautifyParams { skin_smooth: 999.0, saturation: -999.0, ..BeautifyParams::default() }.normalized();
        assert_eq!(p.skin_smooth, 100.0);
        assert_eq!(p.saturation, -100.0);
        assert!(BeautifyParams::default().normalized().is_identity());
        assert!(BeautifyParams { texture_keep: 100.0, ..BeautifyParams::default() }.is_identity());
        assert!(!BeautifyParams { skin_smooth: 1.0, ..BeautifyParams::default() }.is_identity());
        assert_eq!(BeautifyParams::default().eye_enlarge, 0.0);
    }

    #[test]
    fn validate_rejects_bad_values() {
        assert!(BeautifyParams::default().validate().is_ok());
        let nan = BeautifyParams { sharpen: f32::NAN, ..BeautifyParams::default() };
        assert!(nan.validate().is_err());
        let over = BeautifyParams { brighten: 101.0, ..BeautifyParams::default() };
        assert!(over.validate().is_err());
        // normalized() repairs instead of rejecting
        assert!(nan.normalized().validate().is_ok());
    }

    #[test]
    fn presets_exist_differ_and_validate() {
        assert_eq!(BUILTIN_ORDER, ["Natural", "Cosplay", "Editorial"]);
        let natural = builtin("Natural").unwrap();
        let cosplay = builtin("Cosplay").unwrap();
        let editorial = builtin("Editorial").unwrap();
        assert_eq!(natural.eye_enlarge, 0.0);
        assert_ne!(natural, cosplay);
        for p in [&natural, &cosplay, &editorial] {
            assert!(p.validate().is_ok());
        }
        assert!(builtin("Nope").is_err());
    }

    #[test]
    fn preset_json_roundtrip() {
        let p = builtin("Cosplay").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        let back: BeautifyParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
