"""UV preparation, sparse PBR sampling, and glTF material assembly."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from flex_gemm.ops.grid_sample import grid_sample_3d

from .mesh import UVAtlas, unwrap_mesh, validate_mesh
from .raster import rasterize_metal


@dataclass(frozen=True)
class PreparedSurface:
    atlas: UVAtlas
    original_vertices: np.ndarray
    original_faces: np.ndarray
    simplified: bool


def prepare_surface(vertices, faces, *, texture_size: int, decimation_target: int) -> PreparedSurface:
    original_vertices, original_faces = validate_mesh(vertices, faces)
    work_vertices, work_faces = original_vertices, original_faces
    if decimation_target <= 0:
        raise ValueError("decimation_target must be positive")
    # CuMesh's public target is a vertex budget. Triangle surfaces generally
    # approach two faces per vertex, while fast-simplification accepts faces.
    face_target = decimation_target * 2
    simplified = len(work_faces) > face_target
    if simplified:
        import fast_simplification

        work_vertices, work_faces = fast_simplification.simplify(
            work_vertices, work_faces, target_count=face_target
        )
        work_vertices, work_faces = validate_mesh(work_vertices, work_faces)
    atlas = unwrap_mesh(work_vertices, work_faces, texture_size=texture_size)
    return PreparedSurface(atlas, original_vertices, original_faces, simplified)


def atlas_from_trimesh(mesh: trimesh.Trimesh, *, texture_size: int,
                       uv_policy: str = "preserve") -> UVAtlas:
    vertices, faces = validate_mesh(mesh.vertices, mesh.faces)
    if uv_policy not in {"preserve", "regenerate"}:
        raise ValueError("uv_policy must be 'preserve' or 'regenerate'")
    source_uv = getattr(mesh.visual, "uv", None)
    if uv_policy == "preserve" and source_uv is not None:
        uvs = np.ascontiguousarray(source_uv, dtype=np.float32)
        if uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all():
            raise ValueError("source UVs must be finite and have shape [V, 2]")
        return UVAtlas(vertices, faces, uvs, np.arange(len(vertices), dtype=np.uint32))
    return unwrap_mesh(vertices, faces, texture_size=texture_size)


def _project_to_original(points: np.ndarray, surface: PreparedSurface,
                         chunk_size: int = 65_536) -> np.ndarray:
    if not surface.simplified:
        return points
    original = trimesh.Trimesh(
        vertices=surface.original_vertices, faces=surface.original_faces, process=False
    )
    projected = np.empty_like(points)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        closest, distance, _ = trimesh.proximity.closest_point(original, points[start:end])
        if not np.isfinite(closest).all() or not np.isfinite(distance).all():
            raise RuntimeError("closest-surface projection produced non-finite values")
        projected[start:end] = closest
    return projected


def _channel(attrs: np.ndarray, layout: dict[str, slice], name: str,
             default: float | None = None) -> np.ndarray:
    selection = layout.get(name)
    if selection is None:
        if default is None:
            raise ValueError(f"attribute layout is missing {name!r}")
        return np.full((*attrs.shape[:2], 1), default, dtype=np.float32)
    result = attrs[..., selection]
    if result.ndim == 2:
        result = result[..., None]
    return result


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(values * 255, 0, 255).astype(np.uint8)


def _normals_before_uv_seams(atlas: UVAtlas) -> np.ndarray:
    """Match upstream's compute-normals-then-apply-vertex-map ordering."""
    source_count = int(atlas.vertex_map.max()) + 1
    source_positions = np.zeros((source_count, 3), dtype=np.float32)
    source_positions[atlas.vertex_map] = atlas.positions
    source_faces = atlas.vertex_map[atlas.faces]
    source_mesh = trimesh.Trimesh(
        vertices=source_positions, faces=source_faces, process=False
    )
    return np.ascontiguousarray(
        np.asarray(source_mesh.vertex_normals, dtype=np.float32)[atlas.vertex_map]
    )


def bake_pbr_mesh(
    surface: PreparedSurface,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: dict[str, slice],
    *,
    aabb,
    voxel_size=None,
    grid_size=None,
    texture_size: int,
    alpha_mode: str = "OPAQUE",
) -> tuple[trimesh.Trimesh, dict]:
    if alpha_mode not in {"OPAQUE", "BLEND", "MASK"}:
        raise ValueError("alpha_mode must be OPAQUE, BLEND, or MASK")
    atlas = surface.atlas
    raster = rasterize_metal(
        atlas.positions, atlas.faces, atlas.uvs,
        width=texture_size, height=texture_size,
    )
    mask = raster.face_ids != 0
    if not mask.any():
        raise RuntimeError("UV raster produced no covered texels")
    valid_positions = _project_to_original(raster.positions[mask, :3], surface)

    device = attr_volume.device
    aabb_tensor = torch.as_tensor(aabb, dtype=torch.float32, device=device)
    if aabb_tensor.shape != (2, 3):
        raise ValueError("aabb must have shape [2, 3]")
    if voxel_size is None:
        if grid_size is None:
            raise ValueError("voxel_size or grid_size is required")
        grid = torch.as_tensor(grid_size, dtype=torch.int64, device=device)
        voxel = (aabb_tensor[1] - aabb_tensor[0]) / grid
    else:
        voxel = torch.as_tensor(voxel_size, dtype=torch.float32, device=device)
        if voxel.ndim == 0:
            voxel = voxel.repeat(3)
        grid = ((aabb_tensor[1] - aabb_tensor[0]) / voxel).round().to(torch.int64)
    if voxel.shape != (3,) or grid.shape != (3,) or bool((voxel <= 0).any()):
        raise ValueError("voxel and grid dimensions must be positive 3-vectors")
    if coords.ndim != 2 or coords.shape[1] not in {3, 4}:
        raise ValueError("coords must have shape [N, 3] or [N, 4]")
    batched_coords = (
        coords if coords.shape[1] == 4
        else torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    )
    queries = (
        torch.from_numpy(valid_positions).to(device=device, dtype=torch.float32)
        - aabb_tensor[0]
    ) / voxel
    sampled = grid_sample_3d(
        attr_volume,
        batched_coords,
        torch.Size([1, attr_volume.shape[1], *grid.tolist()]),
        queries.reshape(1, -1, 3),
        mode="trilinear",
    )[0].float().cpu().numpy()
    attrs = np.zeros((texture_size, texture_size, attr_volume.shape[1]), dtype=np.float32)
    attrs[mask] = sampled

    base_color = _to_u8(_channel(attrs, attr_layout, "base_color"))
    metallic = _to_u8(_channel(attrs, attr_layout, "metallic", 0.0))
    roughness = _to_u8(_channel(attrs, attr_layout, "roughness", 1.0))
    alpha = _to_u8(_channel(attrs, attr_layout, "alpha", 1.0))
    inverse = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, inverse, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic[..., 0], inverse, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness[..., 0], inverse, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha[..., 0], inverse, 1, cv2.INPAINT_TELEA)[..., None]
    base_rgba = np.concatenate([base_color, alpha], axis=-1)
    metallic_roughness = np.concatenate(
        [np.zeros_like(metallic), roughness, metallic], axis=-1
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base_rgba),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        # This portable path follows upstream's standard, non-remesh branch.
        doubleSided=True,
    )

    vertices = atlas.positions.copy()
    normals = _normals_before_uv_seams(atlas).copy()
    uvs = atlas.uvs.copy()
    vertices[:, [1, 2]] = np.stack([vertices[:, 2], -vertices[:, 1]], axis=1)
    normals[:, [1, 2]] = np.stack([normals[:, 2], -normals[:, 1]], axis=1)
    uvs[:, 1] = 1 - uvs[:, 1]
    result = trimesh.Trimesh(
        vertices=vertices,
        faces=atlas.faces,
        vertex_normals=normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
    )
    evidence = {
        "raster_backend": raster.backend,
        "texture_size": texture_size,
        "covered_texels": int(mask.sum()),
        "total_texels": int(mask.size),
        "simplified": surface.simplified,
        "pbr_channels": {
            "base_color": "baseColorTexture RGB",
            "alpha": "baseColorTexture A",
            "roughness": "metallicRoughnessTexture G",
            "metallic": "metallicRoughnessTexture B",
        },
        "alpha_mode": alpha_mode,
    }
    return result, evidence
