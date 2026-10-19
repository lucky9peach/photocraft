# PortraitLab.app (renamed build of this fork)

Builds a separately named, separately identified app bundle from this fork's release binary, so it
can sit next to the official PhotoCraft.app without fighting it for file associations:

```sh
cargo build --release -p photocraft --features heif
ICON_OUT=/tmp/pl-icon python3 packaging/macos/portraitlab/icon.py
REPO=$PWD WORK=/tmp/pl-app ICON_OUT=/tmp/pl-icon python3 packaging/macos/portraitlab/bundle.py
ditto /tmp/pl-app/PortraitLab.app /Applications/PortraitLab.app
```

`icon.py` draws the icon (`PortraitLab.icns`), `bundle.py` assembles `Contents/{MacOS,Resources,
Info.plist,PkgInfo}` from `packaging/macos/Info.plist.in` with the name/identifier substituted
(`dev.portraitlab.app`), licences copied in, document types demoted to `Alternate`, and the bundle
ad-hoc signed. The build carries this fork's native `photocraft-beauty` crate: Filter › Beautify.
