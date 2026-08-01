#!/usr/bin/env python3
"""Export deterministic fixtures from the pinned O-Voxel CPU implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "build/trellis2/upstream/o-voxel/src/convert/flexible_dual_grid.cpp"
EXTENSIONS = list((ROOT / "build/trellis2/fdg").glob("kg_trellis2_fdg*.so"))
OUTPUT = ROOT / "Tests/KernelGoblinTrellis2Tests/Fixtures/flexible-dual-grid-ovoxel.json"


def load_extension():
    if len(EXTENSIONS) != 1:
        raise SystemExit("build the pinned O-Voxel CPU extension first")
    spec = importlib.util.spec_from_file_location("kg_trellis2_fdg", EXTENSIONS[0])
    if spec is None or spec.loader is None:
        raise SystemExit("could not load the pinned O-Voxel CPU extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_case(module, name, vertices, faces, grid_size, aabb):
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int32)
    grid = torch.tensor(grid_size, dtype=torch.int32)
    bounds = torch.tensor(aabb, dtype=torch.float32)
    voxel_size = (bounds[1] - bounds[0]) / grid
    local_vertices = (vertices - bounds[0]).contiguous()
    grid_range = torch.stack([torch.zeros_like(grid), grid]).contiguous()
    coords, dual, flags = module.mesh_to_flexible_dual_grid_cpu(
        local_vertices,
        faces.contiguous(),
        voxel_size.contiguous(),
        grid_range,
        1.0,
        0.2,
        1.0e-2,
        False,
    )
    return {
        "name": name,
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "grid_size": grid.tolist(),
        "aabb": bounds.tolist(),
        "coords": coords.tolist(),
        "dual_vertices": dual.tolist(),
        "intersections": flags.to(torch.uint8).tolist(),
    }


def main() -> None:
    module = load_extension()
    cases = [
        export_case(
            module,
            "closed_tetrahedron",
            [[0.08, 0.10, 0.12], [0.91, 0.17, 0.22],
             [0.19, 0.88, 0.25], [0.24, 0.31, 0.93]],
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            [7, 6, 5],
            [[0, 0, 0], [1, 1, 1]],
        ),
        export_case(
            module,
            "open_bent_patch",
            [[-0.43, -0.36, -0.18], [0.41, -0.31, -0.08],
             [0.36, 0.39, 0.24], [-0.38, 0.34, 0.11]],
            [[0, 1, 2], [0, 2, 3]],
            [6, 7, 5],
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        ),
    ]
    payload = {
        "format": "kernelgoblin.flexible-dual-grid-voxelizer.v1",
        "upstream_repository": "https://github.com/microsoft/o-voxel",
        "upstream_revision": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        "upstream_source": "src/convert/flexible_dual_grid.cpp",
        "upstream_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "oracle": "mesh_to_flexible_dual_grid_cpu",
        "face_weight": 1.0,
        "boundary_weight": 0.2,
        "regularization_weight": 1.0e-2,
        "torch_version": torch.__version__,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"wrote {OUTPUT} sha256={digest}")


if __name__ == "__main__":
    main()
