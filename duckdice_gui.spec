# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec for building the Tkinter GUI as a single-file windowed executable
# Usage (from project root, on Windows):
#   py -m pip install -r requirements-build.txt
#   pyinstaller --clean --noconfirm duckdice_gui.spec

block_cipher = None

hiddenimports = [
    # Ensure strategy plugins are bundled (registry relies on imports)
    'strategies.anti_martingale_streak',
    'strategies.fib_loss_cluster',
    'strategies.kelly_capped',
    'strategies.range50_random',
    'strategies.max_wager_flow',
    'strategies.faucet_cashout',
]


a = Analysis(
    ['duckdice_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('strategies/*.py', 'strategies'),
        ('FEATURES_OVERVIEW.md', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='duckdice-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
)
