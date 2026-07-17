"""Build the desktop app bundle with PyInstaller, smoke-test it, zip it.

Usage: uv run python packaging/build_app.py [--dist DIR]
Produces Lunation-<version>-<os>-<arch>.zip next to --dist (default
build-app/). Windows: onedir with Lunation.exe. macOS: Lunation.app.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(argv, **kw):
    print("+", " ".join(map(str, argv)))
    subprocess.run(argv, check=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(ROOT, "build-app"))
    args = ap.parse_args()
    dist = os.path.abspath(args.dist)
    work = os.path.join(dist, "work")

    icon = os.path.join(ROOT, "lunation", "gui", "icon.ico")
    add_data = f"{icon}{os.pathsep}lunation/gui"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--windowed", "--name", "Lunation",
           "--distpath", dist, "--workpath", work,
           "--specpath", work,
           "--add-data", add_data,
           # build-env-only helpers of the astropy hook — keep them out
           # of the shipped bundle
           "--exclude-module", "matplotlib",
           "--exclude-module", "pytest",
           # dist metadata read at runtime: `lunation version` (ours) and
           # the xisf package (self-versions at import)
           "--copy-metadata", "lunation",
           "--copy-metadata", "xisf",
           os.path.join(ROOT, "packaging", "app.py")]
    if sys.platform == "win32":
        cmd += ["--icon", icon]
    run(cmd)

    if sys.platform == "darwin":
        bundle = os.path.join(dist, "Lunation.app")
        exe = os.path.join(bundle, "Contents", "MacOS", "Lunation")
    else:
        bundle = os.path.join(dist, "Lunation")
        exe = os.path.join(bundle, "Lunation.exe"
                           if sys.platform == "win32" else "Lunation")

    # smoke: the bundled Qt/numpy/cv2 stack must load and build the window
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    run([exe, "--smoke"], env=env)
    run([exe, "version"], env=env)

    osname = {"win32": "windows", "darwin": "macos"}.get(
        sys.platform, sys.platform)
    arch = platform.machine().lower().replace("amd64", "x64")
    zip_base = os.path.join(dist, f"Lunation-{version('lunation')}"
                            f"-{osname}-{arch}")
    if sys.platform == "darwin":
        # ditto preserves the .app bundle's structure and permissions
        run(["ditto", "-c", "-k", "--keepParent", bundle,
             zip_base + ".zip"])
    else:
        shutil.make_archive(zip_base, "zip", dist, "Lunation")
    print("bundle:", zip_base + ".zip")


if __name__ == "__main__":
    main()
