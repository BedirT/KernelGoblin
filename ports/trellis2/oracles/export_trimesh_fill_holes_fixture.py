#!/usr/bin/env python3
"""Export small-hole topology from the pinned TRELLIS.2 Trimesh dependency."""

import json
from pathlib import Path

import numpy as np
import trimesh


def repaired(vertices, faces):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.fill_holes()
    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
    }


def main():
    fixture = {
        "format": "KernelGoblin-trimesh-fill-holes-v1",
        "source": {
            "repository": "https://github.com/microsoft/TRELLIS.2",
            "revision": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
            "callsite": "trellis2/pipelines/trellis2_image_to_3d.py:470-475",
            "operation": "trimesh.Trimesh.fill_holes",
            "trimesh_version": trimesh.__version__,
        },
        "triangle_hole": repaired(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 3, 1], [1, 3, 2], [2, 3, 0]],
        ),
        "quad_hole": repaired(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
            [[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]],
        ),
    }
    output = (
        Path(__file__).parents[3]
        / "Tests/KernelGoblinTrellis2Tests/Fixtures/trimesh-fill-holes.json"
    )
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
