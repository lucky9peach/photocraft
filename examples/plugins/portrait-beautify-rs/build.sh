#!/bin/sh
# Builds the Portrait Beautify plug-in.
# Needs the wasm32-unknown-unknown target: `rustup target add wasm32-unknown-unknown`.
set -eu
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
out=target/wasm32-unknown-unknown/release/photocraft_plugin_portrait_beautify.wasm
echo "built $out ($(wc -c < "$out") bytes)"
