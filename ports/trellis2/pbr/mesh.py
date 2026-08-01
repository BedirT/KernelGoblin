"""Validated mesh input and deterministic xatlas UV topology handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UVAtlas:
    positions: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    vertex_map: np.ndarray


def validate_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(faces)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("vertices must be a nonempty [V, 3] array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must be a nonempty triangular [F, 3] array")
    if not np.isfinite(positions).all():
        raise ValueError("vertices must be finite")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("faces must contain integer indices")
    if triangles.min() < 0 or triangles.max() >= len(positions):
        raise ValueError("face index exceeds vertex count")
    extent = np.ptp(positions, axis=0)
    if float(extent.max()) <= np.finfo(np.float32).eps:
        raise ValueError("mesh must have nonzero spatial extent")
    return positions, np.ascontiguousarray(triangles, dtype=np.uint32)


def unwrap_mesh(vertices, faces, *, texture_size: int = 2048, padding: int = 2) -> UVAtlas:
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    positions, triangles = validate_mesh(vertices, faces)
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, triangles)
    pack = xatlas.PackOptions()
    pack.resolution = texture_size
    pack.padding = padding
    pack.bilinear = True
    pack.create_image = False
    atlas.generate(pack_options=pack)
    vertex_map, indices, uvs = atlas[0]
    vertex_map = np.ascontiguousarray(vertex_map, dtype=np.uint32)
    result = UVAtlas(
        positions=np.ascontiguousarray(positions[vertex_map], dtype=np.float32),
        faces=np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.uint32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        vertex_map=vertex_map,
    )
    if len(result.faces) == 0 or len(result.positions) == 0:
        raise RuntimeError("xatlas produced an empty atlas")
    if result.faces.max() >= len(result.positions):
        raise RuntimeError("xatlas produced an invalid face index")
    if not np.isfinite(result.uvs).all() or (result.uvs < 0).any() or (result.uvs > 1).any():
        raise RuntimeError("xatlas produced UVs outside [0, 1]")
    return result
