#!/usr/bin/env python3
"""Run pinned TRELLIS.2 inference pipelines on Apple MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


PORT = Path(__file__).resolve().parent
ROOT = PORT.parents[1]
sys.path.insert(0, str(PORT))

import runtime_env

runtime_env.require_no_cpu_fallback()


DINO_REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"


def verify_checkpoint_access() -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        hf_hub_download(DINO_REPOSITORY, "config.json", revision=DINO_REVISION)
    except GatedRepoError as error:
        pending = "awaiting a review" in str(error)
        if pending:
            message = (
                "Access to Meta's DINOv3 ViT-L/16 checkpoint is authenticated "
                "but still awaiting review by the repository authors. Retry after "
                "Hugging Face reports that access was granted. No TRELLIS weights "
                "were downloaded."
            )
        else:
            message = (
                "TRELLIS.2 requires Meta's gated DINOv3 ViT-L/16 checkpoint. "
                "Request access at https://huggingface.co/"
                f"{DINO_REPOSITORY}, accept its terms, then run `hf auth login` "
                "before retrying. No TRELLIS weights were downloaded."
            )
        raise SystemExit(
            message
        ) from error


def release_component(pipeline, *names: str) -> None:
    import gc
    import torch

    torch.mps.synchronize()
    for name in names:
        model = pipeline.models.pop(name, None)
        if model is not None:
            model.cpu()
            del model
    gc.collect()
    torch.mps.empty_cache()


def unload_component(pipeline, name: str) -> None:
    """Evict a lazy component's weights while retaining its reloadable proxy."""
    import gc
    import torch

    torch.mps.synchronize()
    model = pipeline.models[name]
    if not hasattr(model, "unload"):
        raise RuntimeError(f"component is not reloadable: {name}")
    model.unload()
    gc.collect()
    torch.mps.empty_cache()


def sample_shape_slat_cascade_low_memory(
    pipeline,
    lr_cond,
    cond,
    lr_resolution: int,
    resolution: int,
    coords,
    sampler,
    max_num_tokens: int = 49152,
):
    """Run each cascade substage without retaining the previous model's weights."""
    import torch

    lr_slat = pipeline.sample_shape_slat(
        lr_cond,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        sampler,
    )
    release_component(pipeline, "shape_slat_flow_model_512")

    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(pipeline.device)
    decoder.low_vram = True
    hr_coords = decoder.upsample(lr_slat, upsample_times=4)
    del lr_slat
    unload_component(pipeline, "shape_slat_decoder")

    hr_resolution = resolution
    while True:
        quant_coords = torch.cat(
            [
                hr_coords[:, :1],
                (
                    (hr_coords[:, 1:] + 0.5)
                    / lr_resolution
                    * (hr_resolution // 16)
                ).int(),
            ],
            dim=1,
        )
        coords = quant_coords.unique(dim=0)
        if coords.shape[0] < max_num_tokens or hr_resolution == 1024:
            if hr_resolution != resolution:
                print(
                    "Due to the limited number of tokens, the resolution is "
                    f"reduced to {hr_resolution}."
                )
            break
        hr_resolution -= 128
    del hr_coords, quant_coords

    slat = pipeline.sample_shape_slat(
        cond,
        pipeline.models["shape_slat_flow_model_1024"],
        coords,
        sampler,
    )
    release_component(pipeline, "shape_slat_flow_model_1024")
    return slat, hr_resolution


def run_cascade_low_memory(pipeline, image, args):
    """Upstream cascade sequence with one-way component eviction."""
    import gc
    import torch
    from trellis2.representations import MeshWithVoxel

    if not args.no_preprocess:
        image = pipeline.preprocess_image(image)
    torch.manual_seed(args.seed)
    cond_512 = pipeline.get_cond([image], 512)
    cond_1024 = pipeline.get_cond([image], 1024)
    pipeline.image_cond_model.cpu()
    del pipeline.image_cond_model
    if hasattr(pipeline, "rembg_model"):
        pipeline.rembg_model.cpu()
        del pipeline.rembg_model
    gc.collect()
    torch.mps.empty_cache()

    sampler = {"steps": args.steps} if args.steps is not None else {}
    coords = pipeline.sample_sparse_structure(cond_512, 32, 1, sampler)
    release_component(
        pipeline, "sparse_structure_flow_model", "sparse_structure_decoder"
    )
    resolution = 1024 if args.pipeline_type == "1024_cascade" else 1536
    shape_slat, resolution = sample_shape_slat_cascade_low_memory(
        pipeline,
        cond_512,
        cond_1024,
        512,
        resolution,
        coords,
        sampler,
    )
    tex_slat = pipeline.sample_tex_slat(
        cond_1024,
        pipeline.models["tex_slat_flow_model_1024"],
        shape_slat,
        sampler,
    )
    release_component(pipeline, "tex_slat_flow_model_1024")
    del cond_512, cond_1024, coords
    gc.collect()

    meshes, subs = pipeline.decode_shape_slat(shape_slat, resolution)
    release_component(pipeline, "shape_slat_decoder")
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    release_component(pipeline, "tex_slat_decoder")
    output = []
    for mesh, voxel in zip(meshes, tex_voxels):
        mesh.fill_holes()
        output.append(MeshWithVoxel(
            mesh.vertices,
            mesh.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / resolution,
            coords=voxel.coords[:, 1:],
            attrs=voxel.feats,
            voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
            layout=pipeline.pbr_attr_layout,
        ))
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="512",
    )
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--steps", type=int, help="override all three sampler step counts")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument(
        "--alpha-mode", choices=("OPAQUE", "BLEND", "MASK"), default="OPAQUE",
        help="glTF alpha behavior; OPAQUE preserves pinned upstream semantics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input image does not exist: {args.input}")

    import torch
    from PIL import Image

    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")
    verify_checkpoint_access()
    runtime_env.configure_backends()

    import streaming_loader
    streaming_loader.install(lazy=args.pipeline_type.endswith("_cascade"))
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    model_names = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_decoder",
        "tex_slat_decoder",
    ]
    if args.pipeline_type == "512":
        model_names += ["shape_slat_flow_model_512", "tex_slat_flow_model_512"]
    elif args.pipeline_type == "1024":
        model_names += ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"]
    else:
        model_names += [
            "shape_slat_flow_model_512",
            "shape_slat_flow_model_1024",
            "tex_slat_flow_model_1024",
        ]
    Trellis2ImageTo3DPipeline.model_names_to_load = model_names
    started = time.monotonic()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to(torch.device("mps"))

    sampler = {}
    if args.steps is not None:
        sampler["steps"] = args.steps
    image = Image.open(args.input)
    if args.pipeline_type.endswith("_cascade"):
        # The upstream run() entry point is no_grad; preserve that contract
        # while using our stage-wise cascade implementation.
        with torch.inference_mode():
            meshes = run_cascade_low_memory(pipeline, image, args)
    else:
        meshes = pipeline.run(
            image,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            preprocess_image=not args.no_preprocess,
            sparse_structure_sampler_params=sampler,
            shape_slat_sampler_params=sampler,
            tex_slat_sampler_params=sampler,
        )
    mesh = meshes[0]
    if mesh.vertices.numel() == 0 or mesh.faces.numel() == 0:
        raise RuntimeError("TRELLIS.2 produced empty geometry")
    if not torch.isfinite(mesh.vertices).all():
        raise RuntimeError("TRELLIS.2 produced non-finite vertices")
    if int(mesh.faces.min()) < 0 or int(mesh.faces.max()) >= mesh.vertices.shape[0]:
        raise RuntimeError("TRELLIS.2 produced out-of-range face indices")
    args.output.mkdir(parents=True, exist_ok=True)
    glb_path = args.output / f"trellis2-{args.pipeline_type}.glb"
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        alpha_mode=args.alpha_mode,
    )
    pbr_evidence = glb.metadata.get("kernel_goblin_pbr")
    if pbr_evidence is None:
        raise RuntimeError("TRELLIS.2 export did not execute the PBR bake path")
    glb.export(glb_path)
    import trimesh

    reloaded = trimesh.load(glb_path, force="mesh", process=False)
    if len(reloaded.vertices) == 0 or len(reloaded.faces) == 0:
        raise RuntimeError("exported GLB did not reload with non-empty geometry")
    if getattr(reloaded.visual, "uv", None) is None:
        raise RuntimeError("exported GLB did not reload with UV coordinates")
    material = getattr(reloaded.visual, "material", None)
    if material is None or material.baseColorTexture is None:
        raise RuntimeError("exported GLB is missing its base-color texture")
    if material.metallicRoughnessTexture is None:
        raise RuntimeError("exported GLB is missing its metallic-roughness texture")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    effective_steps = args.steps if args.steps is not None else 12
    import resource

    evidence = {
        "backend": "mps",
        "cpu_fallback_enabled": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] != "0",
        "pipeline": args.pipeline_type,
        "seed": args.seed,
        "sampler_steps": effective_steps,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "alpha_mode": args.alpha_mode,
        "preprocess_image": not args.no_preprocess,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "reloaded_vertices": int(len(reloaded.vertices)),
        "reloaded_faces": int(len(reloaded.faces)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_resident_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "output": str(glb_path.resolve()),
        "output_bytes": glb_path.stat().st_size,
        "output_sha256": sha256(glb_path),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "revisions": {
            "trellis2_source": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
            "trellis2_weights": "af44b45f2e35a493886929c6d786e563ec68364d",
            "trellis_image_large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            "dinov3": DINO_REVISION,
        },
        "export": "UV-mapped PBR GLB with Metal rasterization",
        "pbr": pbr_evidence,
    }
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
