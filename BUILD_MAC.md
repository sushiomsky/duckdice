### macOS packaging (.app and .dmg) — Build and Install Guide

This project includes a simple macOS build script that uses PyInstaller to produce:
- A standalone GUI app bundle: `duckdice-gui.app`
- A standalone CLI binary: `duckdice`
- A distributable DMG image containing the GUI app and a link to Applications

Tested on macOS 12+ (Monterey, Ventura, Sonoma) with Apple Silicon (arm64) and Intel (x86_64). Build on the architecture you target. Intel builds run on Apple Silicon under Rosetta if enabled.

#### Prerequisites
- Xcode Command Line Tools installed (for `hdiutil` and basic toolchain):
  ```bash
  xcode-select --install
  ```
- Python 3.9+ and pip
- PyInstaller from `requirements-build.txt`:
  ```bash
  python3 -m pip install -r requirements-build.txt
  ```

Optional (for signing/notarization): an Apple Developer ID Application certificate and App-Specific Password.

#### Quick build
From project root:
```bash
chmod +x build/macos/build.sh
./build/macos/build.sh --clean --version 1.0.0
```
Outputs:
- App bundle: `dist/duckdice-gui/duckdice-gui.app`
- CLI binary: `dist/duckdice/duckdice`
- DMG: `dist/macos/duckdice-gui-1.0.0.dmg`

If you omit `--version`, it defaults to `0.0.0-dev`.

#### Run locally
- GUI: double-click `duckdice-gui.app`
- CLI: from Terminal `./dist/duckdice/duckdice --help`

If macOS warns that the app is from an unidentified developer, right-click the app and choose Open (Gatekeeper). For distribution to wider audiences, consider code-signing and notarization (see below).

#### Code signing and notarization (optional)
The script includes commented commands showing where to sign. Replace placeholders:
- TEAM_ID: your Apple Developer Team ID
- IDENTITY: "Developer ID Application: Your Name (TEAMID)"
- BUNDLE_ID: e.g., io.duckdice.betbot.gui
- NOTARIZE credentials: use Xcode notarytool or altool (deprecated) with an API key or Apple ID.

Example sign commands (uncomment and set variables in the script):
```bash
codesign --deep --force --options runtime --sign "$IDENTITY" dist/duckdice-gui/duckdice-gui.app
xcrun notarytool submit dist/macos/duckdice-gui-1.0.0.dmg --keychain-profile "AC_PROFILE" --wait
xcrun stapler staple dist/duckdice-gui/duckdice-gui.app
```

#### Troubleshooting
- If PyInstaller misses strategy plugins, ensure imports in `autobet_engine.py` remain present. The `.spec` files already include hidden imports.
- If Tkinter is missing: macOS python.org installers ship Tk; Homebrew Python sometimes needs `brew install tcl-tk` and environment config. Since we ship a bundled app, end-users usually won't need Tk installed.
- If Gatekeeper blocks the app, see the signing/notarization notes above.
