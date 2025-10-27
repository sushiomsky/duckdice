Windows build and distribution guide (PyInstaller)

This project includes PyInstaller specs and a PowerShell build script to produce standalone .exe binaries for both the CLI and the Tkinter GUI.

What gets built
- CLI: dist\duckdice\duckdice.exe
- GUI: dist\duckdice-gui\duckdice-gui.exe

Prerequisites
- Windows 10/11 with Python 3.9+ installed from python.org (or Microsoft Store). Ensure the Python Launcher `py` is available.
- PowerShell with permission to run local scripts (see below).
- Internet access to fetch pip packages during build.

Important notes about tkinter
- Do NOT pip-install `tkinter`. It ships with the standard Python installer on Windows. If you installed a minimal/embedded Python that lacks tkinter, reinstall Python from python.org.

Quick build (PowerShell)
1) Open PowerShell in the project root (folder containing duckdice.py).
2) Allow running the local script for this session:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
3) Install build tool and run build:
   .\build\windows\build.ps1 -Clean

This will:
- Upgrade pip, install PyInstaller, and ensure `requests` is present.
- Build both executables using the provided spec files: `duckdice.spec` (console) and `duckdice_gui.spec` (windowed).
- Place outputs under `dist\\duckdice\\` and `dist\\duckdice-gui\\`.

Building manually (without the script)
- Install build dependencies:
  py -3 -m pip install --upgrade pip wheel setuptools
  py -3 -m pip install -r requirements-build.txt
- Build the CLI exe:
  py -3 -m PyInstaller --clean --noconfirm duckdice.spec
- Build the GUI exe:
  py -3 -m PyInstaller --clean --noconfirm duckdice_gui.spec

Running the executables
- CLI:
  .\dist\duckdice\duckdice.exe --api-key YOUR_KEY user-info
  .\dist\duckdice\duckdice.exe --api-key YOUR_KEY auto-bet --symbol BTC --list-strategies
- GUI:
  .\dist\duckdice-gui\duckdice-gui.exe

WSL users
- Build the Windows .exe from Windows Python, not inside WSL. You can keep your source in the WSL filesystem but open a Windows PowerShell and run the script pointing at the project path, or place the repo on a Windows drive (e.g., C:\path\to\betbot).

Customizing builds
- Spec files: `duckdice.spec` (console app) and `duckdice_gui.spec` (windowed). They already include hidden imports for all strategy plugins and bundle some docs as data.
- To change the exe names, edit the `name` field inside each spec file.
- To include additional data files, add them to the `datas=[...]` list in the spec.

Versioned output folders
- The build script supports an optional -Version parameter to rename the dist folder with a tag and timestamp:
  .\build\windows\build.ps1 -Clean -Version v1.2.3

Troubleshooting
- Antivirus flags or quarantines the exe:
  - This happens sometimes with new unsigned executables. Try excluding the dist folder or sign the binary (see below).
- Missing VCRUNTIME or MSVCP DLLs:
  - Ensure you built using a standard Python from python.org; PyInstaller bundles needed runtime components.
- GUI fails to launch with a tkinter error:
  - Your Python environment may not include tkinter. Reinstall Python from python.org. You do NOT need to pip-install tkinter.
- SSL or TLS errors when calling the API:
  - Ensure the system root certificates are up to date. Running Windows Update often resolves this.

Code signing (optional)
- For distribution to other machines, consider code-signing the exe to reduce SmartScreen warnings. PyInstaller can be combined with signtool.exe after build:
  signtool sign /tr http://timestamp.digicert.com /td SHA256 /fd SHA256 /a dist\duckdice\duckdice.exe
  signtool sign /tr http://timestamp.digicert.com /td SHA256 /fd SHA256 /a dist\duckdice-gui\duckdice-gui.exe

Reproducibility
- PyInstaller versions and Python versions affect output. The project pins PyInstaller in `requirements-build.txt`. If you need deterministic artifacts, build inside a dedicated Windows VM with fixed versions.
