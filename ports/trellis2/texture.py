#!/usr/bin/env python3
"""Texture an existing mesh with TRELLIS.2 on Apple MPS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path


PORT = Path(__file__).resolve().parent
ROOT = PORT.parents[1]
sys.path.insert(0, str(PORT))

import runtime_env

runtime_env.require_no_cpu_fallback()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(name: str, started: float) -> None:
    print(json.dumps({"stage": name, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)


def release_lazy(pipeline, name: str) -> None:
    import torch

    torch.mps.synchronize()
    model = pipeline.models[name]
    if not hasattr(model, "unload"):
        raise RuntimeError(f"component is not a reloadable stage: {name}")
    model.unload()
    gc.collect()
    torch.mps.empty_cache()


def load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("input must resolve to a triangular mesh")
    return loaded


def normalize_mesh(mesh):
    import numpy as np
    import trimesh
    from pbr import validate_mesh

    vertices, faces = validate_mesh(mesh.vertices, mesh.faces)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = float((maximum - minimum).max())
    if extent <= np.finfo(np.float32).eps:
        raise ValueError("mesh must have nonzero spatial extent")
    vertices = (vertices - (minimum + maximum) * 0.5) * (0.99999 / extent)
    vertices[:, [1, 2]] = np.stack([-vertices[:, 2], vertices[:, 1]], axis=1)
    source_uv = getattr(mesh.visual, "uv", None)
    visual = None
    if source_uv is not None:
        uvs = np.ascontiguousarray(source_uv, dtype=np.float32)
        if uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all():
            raise ValueError("source UVs must be finite and have shape [V, 2]")
        visual = trimesh.visual.TextureVisuals(uv=uvs.copy())
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visual, process=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="conditioning image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=512)
    parser.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--uv-policy", choices=("preserve", "regenerate"), default="preserve")
    parser.add_argument("--alpha-mode", choices=("OPAQUE", "BLEND", "MASK"), default="OPAQUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mesh.is_file():
        raise SystemExit(f"input mesh does not exist: {args.mesh}")
    if not args.input.is_file():
        raise SystemExit(f"conditioning image does not exist: {args.input}")
    if args.steps <= 0:
        raise SystemExit("steps must be positive")

    import torch
    import trimesh
    from PIL import Image

    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")
    from run import verify_checkpoint_access

    verify_checkpoint_access()
    runtime_env.configure_backends()
    import streaming_loader
    from pbr import PreparedSurface, atlas_from_trimesh, bake_pbr_mesh

    streaming_loader.install(lazy=True)
    from trellis2.pipelines import Trellis2TexturingPipeline

    flow_name = (
        "tex_slat_flow_model_512" if args.resolution == 512
        else "tex_slat_flow_model_1024"
    )
    Trellis2TexturingPipeline.model_names_to_load = [
        "shape_slat_encoder", flow_name, "tex_slat_decoder"
    ]
    started = time.monotonic()
    source_mesh = load_mesh(args.mesh)
    mesh = normalize_mesh(source_mesh)
    atlas = atlas_from_trimesh(
        mesh, texture_size=args.texture_size, uv_policy=args.uv_policy
    )
    surface = PreparedSurface(
        atlas=atlas,
        original_vertices=atlas.positions,
        original_faces=atlas.faces,
        simplified=False,
    )
    with Image.open(args.input) as source_image:
        image = source_image.copy()

    pipeline = Trellis2TexturingPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", "texturing_pipeline.json"
    )
    pipeline.to(torch.device("mps"))
    if not args.no_preprocess:
        image = pipeline.preprocess_image(image)
    if getattr(pipeline.rembg_model, "model", None) is not None:
        pipeline.rembg_model.cpu()
        del pipeline.rembg_model.model
        pipeline.rembg_model.model = None
        gc.collect()
        torch.mps.empty_cache()
    stage("inputs_prepared", started)

    torch.manual_seed(args.seed)
    cond_resolution = 512 if args.resolution == 512 else 1024
    cond = pipeline.get_cond([image], cond_resolution)
    pipeline.image_cond_model.cpu()
    del pipeline.image_cond_model
    gc.collect()
    torch.mps.empty_cache()
    stage("image_conditioned", started)

    pipeline.low_vram = False
    pipeline.models["shape_slat_encoder"].to(torch.device("mps"))
    shape_slat = pipeline.encode_shape_slat(mesh, args.resolution)
    release_lazy(pipeline, "shape_slat_encoder")
    stage("shape_encoded", started)

    pipeline.models[flow_name].to(torch.device("mps"))
    tex_slat = pipeline.sample_tex_slat(
        cond, pipeline.models[flow_name], shape_slat,
        {"steps": args.steps},
    )
    release_lazy(pipeline, flow_name)
    del cond, shape_slat
    gc.collect()
    torch.mps.empty_cache()
    stage("texture_sampled", started)

    pipeline.models["tex_slat_decoder"].to(torch.device("mps"))
    pbr_voxel = pipeline.decode_tex_slat(tex_slat)
    release_lazy(pipeline, "tex_slat_decoder")
    del tex_slat
    gc.collect()
    torch.mps.empty_cache()
    stage("pbr_decoded", started)

    output_mesh, pbr_evidence = bake_pbr_mesh(
        surface,
        pbr_voxel.feats,
        pbr_voxel.coords,
        pipeline.pbr_attr_layout,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=[args.resolution] * 3,
        texture_size=args.texture_size,
        alpha_mode=args.alpha_mode,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    glb_path = args.output / "trellis2-textured.glb"
    output_mesh.export(glb_path)
    reloaded = trimesh.load(glb_path, force="mesh", process=False)
    if not len(reloaded.vertices) or not len(reloaded.faces):
        raise RuntimeError("textured GLB reloaded with empty geometry")
    if getattr(reloaded.visual, "uv", None) is None:
        raise RuntimeError("textured GLB did not retain UV coordinates")
    material = getattr(reloaded.visual, "material", None)
    if material is None or material.baseColorTexture is None:
        raise RuntimeError("textured GLB is missing its base-color texture")
    if material.metallicRoughnessTexture is None:
        raise RuntimeError("textured GLB is missing its metallic-roughness texture")
    stage("glb_verified", started)

    evidence = {
        "backend": "mps",
        "cpu_fallback_enabled": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] != "0",
        "task": "existing_mesh_texturing",
        "seed": args.seed,
        "resolution": args.resolution,
        "texture_size": args.texture_size,
        "sampler_steps": args.steps,
        "uv_policy": args.uv_policy,
        "source_uv_present": getattr(source_mesh.visual, "uv", None) is not None,
        "preprocess_image": not args.no_preprocess,
        "mesh_input": str(args.mesh.resolve()),
        "mesh_input_sha256": sha256(args.mesh),
        "image_input": str(args.input.resolve()),
        "image_input_sha256": sha256(args.input),
        "vertices": int(len(reloaded.vertices)),
        "faces": int(len(reloaded.faces)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_resident_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "output": str(glb_path.resolve()),
        "output_bytes": glb_path.stat().st_size,
        "output_sha256": sha256(glb_path),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "pbr": pbr_evidence,
        "revisions": {
            "trellis2_source": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
            "trellis2_weights": "af44b45f2e35a493886929c6d786e563ec68364d",
            "dinov3": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
            "eigen": "21e4582d1739107337a03460c81412981130373e",
        },
    }
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
