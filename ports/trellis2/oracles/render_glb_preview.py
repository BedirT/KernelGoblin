#!/usr/bin/env python3
"""Render a deterministic three-view preview from a PBR GLB.

This is an optional documentation/oracle tool. It is not used by the native
runtime and intentionally lives in the isolated TRELLIS Python environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    around_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    around_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return around_x @ around_y


def load_geometry(path: Path):
    scene = trimesh.load(path, force="scene", process=False)
    if len(scene.geometry) != 1:
        raise SystemExit("preview renderer requires one GLB mesh")
    mesh = next(iter(scene.geometry.values()))
    if mesh.visual.uv is None or mesh.visual.material.baseColorTexture is None:
        raise SystemExit("preview renderer requires a textured PBR mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
    texture = np.asarray(
        mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64
    )
    height, width = texture.shape[:2]
    x = np.clip(np.rint(uvs[:, 0] * (width - 1)), 0, width - 1).astype(np.int64)
    y = np.clip(np.rint((1 - uvs[:, 1]) * (height - 1)), 0, height - 1).astype(np.int64)
    vertex_colors = texture[y, x]
    face_colors = vertex_colors[faces].mean(axis=1)
    return vertices, faces, face_colors


def render_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    yaw: float,
    pitch: float,
    size: int,
) -> Image.Image:
    transform = rotation(yaw, pitch)
    rotated = vertices @ transform.T
    center = (rotated.min(axis=0) + rotated.max(axis=0)) * 0.5
    rotated -= center
    extent = np.maximum(np.ptp(rotated[:, :2], axis=0), 1e-12)
    scale = (size * 0.82) / extent.max()
    projected = np.empty((vertices.shape[0], 2), dtype=np.float64)
    projected[:, 0] = rotated[:, 0] * scale + size * 0.5
    projected[:, 1] = -rotated[:, 1] * scale + size * 0.52

    triangles = rotated[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-20
    normals[valid] /= lengths[valid, None]
    light = np.array([-0.35, 0.55, 0.76])
    light /= np.linalg.norm(light)
    shading = 0.58 + 0.42 * np.clip(np.abs(normals @ light), 0, 1)
    colors = np.clip(face_colors * (1.35 * shading[:, None]), 0, 255).astype(np.uint8)
    depth = triangles[:, :, 2].mean(axis=1)
    order = np.argsort(depth)

    panel = Image.new("RGB", (size, size), (13, 18, 24))
    draw = ImageDraw.Draw(panel)
    for face_index in order:
        points = [tuple(value) for value in projected[faces[face_index]]]
        color = tuple(int(value) for value in colors[face_index])
        draw.polygon(points, fill=color)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--panel-size", type=int, default=640)
    args = parser.parse_args()
    if args.panel_size < 128 or args.panel_size > 2048:
        raise SystemExit("--panel-size must be between 128 and 2048")
    vertices, faces, colors = load_geometry(args.input)
    views = [(30, -12, "front"), (150, -12, "back"), (270, -4, "side")]
    margin = 20
    title_height = 44
    width = len(views) * args.panel_size + (len(views) + 1) * margin
    height = args.panel_size + title_height + margin * 2
    result = Image.new("RGB", (width, height), (8, 12, 17))
    draw = ImageDraw.Draw(result)
    for index, (yaw, pitch, label) in enumerate(views):
        panel = render_panel(
            vertices, faces, colors, yaw=yaw, pitch=pitch,
            size=args.panel_size,
        )
        x = margin + index * (args.panel_size + margin)
        result.paste(panel, (x, title_height + margin))
        draw.text((x + 8, 14), label.upper(), fill=(190, 204, 218))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output, optimize=True)
    print(
        f"wrote {args.output} views={len(views)} "
        f"vertices={vertices.shape[0]} faces={faces.shape[0]}"
    )


if __name__ == "__main__":
    main()
