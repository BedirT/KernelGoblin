"""Portable UV and PBR bake stages for the TRELLIS.2 runtime."""

from .mesh import UVAtlas, unwrap_mesh, validate_mesh
from .raster import RasterResult, rasterize_metal
from .bake import PreparedSurface, atlas_from_trimesh, bake_pbr_mesh, prepare_surface

__all__ = [
    "PreparedSurface", "RasterResult", "UVAtlas", "atlas_from_trimesh",
    "bake_pbr_mesh", "prepare_surface", "rasterize_metal", "unwrap_mesh",
    "validate_mesh",
]
