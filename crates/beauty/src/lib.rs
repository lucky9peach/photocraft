//! Native portrait/body beautification: params, presets, mask math.
//!
//! Stage 1 foundation. Detection, skin and warp stages build on this crate;
//! see `docs/beauty-requirements.md` for the roadmap.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::unimplemented)]
#![deny(clippy::todo)]
#![deny(clippy::unreachable)]

pub mod mask;
pub mod params;

pub use params::{BeautifyParams, BUILTIN_ORDER, PRESETS};

use core::fmt;

/// Errors for fallible beauty operations. Non-test code never panics.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A parameter name or value is invalid.
    InvalidParam(String),
    /// A buffer length does not match its declared geometry.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// An input is empty where at least one element is required.
    EmptyInput(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam(msg) => write!(f, "invalid param: {msg}"),
            Error::ShapeMismatch { what, expected, found } => write!(
                f,
                "shape mismatch for {what}: expected {expected} samples, found {found}"
            ),
            Error::EmptyInput(what) => write!(f, "empty input: {what}"),
        }
    }
}

impl std::error::Error for Error {}
