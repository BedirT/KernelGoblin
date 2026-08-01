#!/usr/bin/env python3
"""Build the pinned CPU-only O-Voxel mesh-to-dual-grid extension."""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "ports" / "trellis2"
OVOXEL = ROOT / "build" / "trellis2" / "upstream" / "o-voxel"
BUILD = ROOT / "build" / "trellis2" / "fdg"


def main() -> None:
    eigen = OVOXEL / "third_party" / "eigen" / "Eigen" / "Core"
    if not eigen.is_file():
        raise SystemExit("pinned Eigen submodule is missing")
    BUILD.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "4")
    module = load(
        name="kg_trellis2_fdg",
        sources=[
            str(PORT / "native" / "fdg_bindings.cpp"),
            str(PORT / "native" / "flexible_dual_grid_cpu.cpp"),
        ],
        extra_include_paths=[str(OVOXEL / "src"), str(OVOXEL / "third_party" / "eigen")],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(BUILD),
        verbose=True,
        is_python_module=True,
    )
    print(f"PASS: loaded {module.__name__} from {module.__file__}")


if __name__ == "__main__":
    main()
