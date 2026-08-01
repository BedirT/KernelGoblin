#!/usr/bin/env python3
"""Prepare a pinned, isolated TRELLIS.2 MPS runtime under build/."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "ports" / "trellis2"
BUILD = ROOT / "build" / "trellis2"
VENV = BUILD / ".venv"
UPSTREAM = BUILD / "upstream"
REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_checkout() -> None:
    if not (UPSTREAM / ".git").is_dir():
        BUILD.mkdir(parents=True, exist_ok=True)
        run(
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/microsoft/TRELLIS.2.git", str(UPSTREAM),
        )
        run("git", "checkout", REVISION, cwd=UPSTREAM)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if actual != REVISION:
        raise SystemExit(f"runtime checkout is {actual}, expected {REVISION}")
    run(
        "git", "submodule", "update", "--init", "--depth", "1",
        "o-voxel/third_party/eigen", cwd=UPSTREAM,
    )

    source = PORT / "overlays" / "conv_mps.py"
    destination = UPSTREAM / "trellis2" / "modules" / "sparse" / "conv" / "conv_mps.py"
    shutil.copy2(source, destination)


def prepare_environment() -> None:
    if not (VENV / "bin" / "python").is_file():
        run("uv", "venv", str(VENV), "--python", "3.11")
    run(
        "uv", "pip", "install", "--python", str(VENV / "bin" / "python"),
        "-r", str(PORT / "requirements.txt"),
    )
    run("cmake", "--preset", "trellis2-uv-raster")
    run("cmake", "--build", "--preset", "trellis2-uv-raster")
    run(str(VENV / "bin" / "python"), str(PORT / "build_fdg_extension.py"))


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("the trellis2 MPS runtime requires macOS")
    prepare_checkout()
    prepare_environment()
    print(f"PASS: TRELLIS.2 MPS runtime prepared at {BUILD}")


if __name__ == "__main__":
    main()
