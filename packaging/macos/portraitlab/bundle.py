#!/usr/bin/env python3
"""Assemble PortraitLab.app from the built release binary (PhotoCraft fork + native beauty)."""
import os, shutil, subprocess, sys, plistlib
from pathlib import Path

ROOT = Path(os.environ["REPO"])
WORK = Path(os.environ["WORK"])
ICON = Path(os.environ["ICON_OUT"]) / "PortraitLab.icns"
APP = WORK / "PortraitLab.app"
BIN = ROOT / "target/release/photocraft"
SHA = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
VERSION = "0.9.0"

if not BIN.is_file():
    sys.exit(f"missing binary: {BIN}")
if APP.exists():
    shutil.rmtree(APP)
for d in ["Contents/MacOS", "Contents/Resources/Licenses"]:
    (APP / d).mkdir(parents=True)

shutil.copy2(BIN, APP / "Contents/MacOS/PortraitLab")
shutil.copy2(ICON, APP / "Contents/Resources/PortraitLab.icns")
for lic in ["LICENSE-MIT", "LICENSE-APACHE", "NOTICE", "ATTRIBUTION.md"]:
    if (ROOT / lic).is_file():
        shutil.copy2(ROOT / lic, APP / "Contents/Resources/Licenses" / lic)

plist = (ROOT / "packaging/macos/Info.plist.in").read_text()
plist = plist.replace("@VERSION@", VERSION).replace("@SHORT_VERSION@", VERSION).replace("@BUILD_SHA@", SHA)
plist = plist.replace("<string>PhotoCraft</string>", "<string>PortraitLab</string>")
plist = plist.replace("<string>ai.storyteller.photocraft</string>", "<string>dev.portraitlab.app</string>")
# Never steal the official app's file associations: register as an alternative handler.
plist = plist.replace("<string>Owner</string>", "<string>Alternate</string>")
plist = plist.replace("Copyright © the PhotoCraft authors.", "PortraitLab build of PhotoCraft (© the PhotoCraft authors).")
(APP / "Contents/Info.plist").write_text(plist)
(APP / "Contents/PkgInfo").write_text("APPL????")

# Validate the plist parses and carries the renamed identity.
with open(APP / "Contents/Info.plist", "rb") as fh:
    d = plistlib.load(fh)
for key, want in [("CFBundleName", "PortraitLab"), ("CFBundleDisplayName", "PortraitLab"),
                  ("CFBundleExecutable", "PortraitLab"), ("CFBundleIdentifier", "dev.portraitlab.app"),
                  ("CFBundleIconFile", "PortraitLab"), ("CFBundleShortVersionString", VERSION)]:
    got = d.get(key)
    if got != want:
        sys.exit(f"plist {key} = {got!r}, expected {want!r}")
ranks = [t.get("LSHandlerRank") for t in d["CFBundleDocumentTypes"]]
if "Owner" in ranks:
    sys.exit("a document type still claims Owner")
print(f"plist OK: {d['CFBundleName']} {d['CFBundleShortVersionString']} ({d['PhotoCraftBuildCommit'][:8]}), handler ranks {ranks}")

ent = ROOT / "packaging/macos/entitlements.plist"
for target in [APP / "Contents/MacOS/PortraitLab", APP]:
    subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", "--options", "runtime",
                    "--entitlements", str(ent), str(target)], check=True)
subprocess.run(["codesign", "--verify", "--strict", "--deep", "--verbose=2", str(APP)], check=True)
print("signed (ad-hoc) and verified:", APP)
