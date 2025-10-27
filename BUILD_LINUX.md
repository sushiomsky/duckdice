### Linux packaging (.deb) — Build and Install Guide

This project ships simple scripts to build Debian packages for Ubuntu/Debian using PyInstaller outputs. You will get two packages:
- duckdice-cli: Installs the CLI binary `duckdice` into `/usr/bin`.
- duckdice-gui: Installs the GUI binary `duckdice-gui` into `/usr/bin` and a desktop entry.

Both packages place the actual binaries under `/opt/duckdice/`.

#### Prerequisites
- A Debian/Ubuntu x86_64 (amd64) or arm64 machine (build on the same architecture you target)
- Python 3.9+ and pip
- PyInstaller (installed via `requirements-build.txt`), and Debian tools:
  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential python3 python3-venv python3-pip fakeroot dpkg-dev debhelper
  ```

#### Quick build
From the project root:
```bash
chmod +x build/linux/build.sh
./build/linux/build.sh --clean --version 1.0.0
```
Outputs:
- `dist/linux/deb/duckdice-cli_1.0.0_amd64.deb` (or `_arm64.deb`)
- `dist/linux/deb/duckdice-gui_1.0.0_amd64.deb` (or `_arm64.deb`)

If you omit `--version`, it defaults to `0.0.0-dev` and uses the current date/time as a build stamp in the control file.

#### Install the packages
```bash
sudo apt-get install -y ./dist/linux/deb/duckdice-cli_*_amd64.deb
sudo apt-get install -y ./dist/linux/deb/duckdice-gui_*_amd64.deb
```
Replace `amd64` with `arm64` if appropriate.

This will install:
- Binaries: `/usr/bin/duckdice` and `/usr/bin/duckdice-gui`
- Installed payloads under `/opt/duckdice/`
- Desktop file: `/usr/share/applications/duckdice-gui.desktop`
- Icon (if present): `/usr/share/icons/hicolor/256x256/apps/duckdice.png`

#### Uninstall
```bash
sudo apt-get remove duckdice-cli duckdice-gui
```

#### Notes
- The build script uses the project’s PyInstaller spec files (`duckdice.spec`, `duckdice_gui.spec`). It runs PyInstaller on Linux to produce native ELF binaries, then wraps them as `.deb` packages.
- If you want to include a custom icon, place a 256x256 PNG at `assets/icons/duckdice.png` before running the build. The script will detect and include it.
- To target a different architecture, build on that architecture (or use a Docker/VM for cross builds).
- For headless servers, the GUI app still builds, but running it requires a graphical environment.

#### Troubleshooting
- If PyInstaller misses strategy plugins, ensure imports in `autobet_engine.py` remain present so they’re collected. The spec files already include hiddenimports when needed.
- If you see `module not found` inside the packaged app, try a clean build: `./build/linux/build.sh --clean`.
- For Ubuntu without Tkinter, remember this repo bundles a GUI executable; you do not need `python3-tk` at runtime when using the packaged binary.
