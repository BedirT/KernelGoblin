#!/usr/bin/env python3
"""Export the pinned mesh -> O-Voxel -> shape-encoder MPS handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CHECKPOINT_SHA256 = "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41"
SOURCES = {
    "ovoxel": UPSTREAM / "o-voxel/src/convert/flexible_dual_grid.cpp",
    "pipeline": UPSTREAM / "trellis2/pipelines/trellis2_texturing.py",
    "fdg_encoder": UPSTREAM / "trellis2/models/sc_vaes/fdg_vae.py",
    "sparse_encoder": UPSTREAM / "trellis2/models/sc_vaes/sparse_unet_vae.py",
    "spatial2channel": UPSTREAM / "trellis2/modules/sparse/spatial/spatial2channel.py",
    "mps_overlay": PORT / "overlays/conv_mps.py",
    "cpu_extension": PORT / "native/flexible_dual_grid_cpu.cpp",
    "cpu_binding": PORT / "native/fdg_bindings.cpp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_voxelizer():
    extensions = list((ROOT / "build/trellis2/fdg").glob("kg_trellis2_fdg*.so"))
    if len(extensions) != 1:
        raise SystemExit("build the pinned O-Voxel CPU extension first")
    spec = importlib.util.spec_from_file_location("kg_trellis2_fdg", extensions[0])
    if spec is None or spec.loader is None:
        raise SystemExit("could not load the pinned O-Voxel CPU extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip() != REVISION or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("upstream or shape encoder checkpoint is not pinned")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is unavailable")
    installed_overlay = (
        UPSTREAM / "trellis2/modules/sparse/conv/conv_mps.py"
    )
    if sha256(installed_overlay) != sha256(SOURCES["mps_overlay"]):
        raise SystemExit("installed MPS sparse-convolution overlay is not pinned")

    sys.path.insert(0, str(PORT / "shims"))
    sys.path.insert(1, str(PORT))
    sys.path.insert(2, str(UPSTREAM))
    os.environ["SPARSE_CONV_BACKEND"] = "mps"
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config as sparse_config
    from trellis2.modules.sparse.spatial.spatial2channel import SparseSpatial2Channel
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder

    sparse_config.set_conv_backend("mps")
    voxelizer = load_voxelizer()
    grid = torch.tensor([512, 512, 512], dtype=torch.int32)
    bounds = torch.tensor([[-0.5] * 3, [0.5] * 3], dtype=torch.float32)
    voxel_vertices = torch.tensor(
        [
            [250.2, 250.2, 250.2],
            [251.7, 250.4, 250.5],
            [250.4, 251.7, 250.6],
            [250.5, 250.7, 251.7],
        ],
        dtype=torch.float32,
    )
    vertices = voxel_vertices / 512.0 - 0.5
    faces = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32
    )
    voxel_size = (bounds[1] - bounds[0]) / grid
    coords, dual, flags = voxelizer.mesh_to_flexible_dual_grid_cpu(
        (vertices - bounds[0]).contiguous(), faces.contiguous(),
        voxel_size.contiguous(),
        torch.stack([torch.zeros_like(grid), grid]).contiguous(),
        1.0, 0.2, 1.0e-2, False,
    )
    sparse_coords = torch.cat(
        [torch.zeros_like(coords[:, :1]), coords], dim=1
    )
    raw_vertex_features = dual / voxel_size - coords

    device = torch.device("mps")
    encoder = FlexiDualGridVaeEncoder(
        model_channels=[64, 128, 256, 512, 1024], latent_channels=32,
        num_blocks=[0, 4, 8, 16, 4],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        down_block_type=["SparseResBlockS2C3d"] * 4,
        block_args=[{}] * 5, use_fp16=True,
    )
    encoder.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    encoder.to(device).eval()
    centered_input: list[torch.Tensor] = []
    guide_records: dict[int, dict[str, torch.Tensor]] = {}

    def capture_input(_module, inputs):
        centered_input.append(inputs[0].feats.detach())

    def capture_guide(module, inputs, output):
        key = id(module)
        if key in guide_records:
            return
        source = inputs[0]
        _coarse, parents, children = source.get_spatial_cache("spatial2channel_2")
        guide_records[key] = {
            "fine": source.coords.detach(),
            "coarse": output.coords.detach(),
            "parents": parents.detach(),
            "children": children.detach(),
        }

    hooks = [encoder.input_layer.register_forward_pre_hook(capture_input)]
    hooks += [
        module.register_forward_hook(capture_guide)
        for module in encoder.modules()
        if isinstance(module, SparseSpatial2Channel)
    ]
    with torch.inference_mode():
        vertex_tensor = sp.SparseTensor(
            raw_vertex_features.to(device), sparse_coords.to(device)
        )
        intersection_tensor = vertex_tensor.replace(flags.to(device))
        latent = encoder(vertex_tensor, intersection_tensor)
    for hook in hooks:
        hook.remove()
    torch.mps.synchronize()
    if len(centered_input) != 1 or len(guide_records) != 4:
        raise SystemExit("shape encoder trace did not capture the expected handoff")

    arrays = [
        ("mesh_vertices", vertices), ("dual_vertices", dual),
        ("centered_input", centered_input[0]), ("latent", latent.feats),
    ]
    payload = bytearray()
    layout = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        layout[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    guides = list(guide_records.values())[::-1]
    metadata = {
        "format": "KernelGoblin mesh to sparse shape encoder MPS oracle v1",
        "source_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "device": "mps",
        "pytorch_enable_mps_fallback": "0",
        "oracle_call": "O-Voxel CPU then FlexiDualGridVaeEncoder.forward on MPS",
        "grid_size": grid.tolist(), "aabb": bounds.tolist(),
        "faces": faces.tolist(), "voxel_coordinates": coords.tolist(),
        "intersections": flags.to(torch.uint8).tolist(),
        "output_coordinates": cpu_list(latent.coords),
        "guides_coarsest_to_finest": [
            {
                "fine_coordinates": cpu_list(value["fine"]),
                "coarse_coordinates": cpu_list(value["coarse"]),
                "parent_indices": cpu_list(value["parents"]),
                "child_indices": cpu_list(value["children"]),
            }
            for value in guides
        ],
        "trace_layout": layout,
        "source_sha256": {
            name: sha256(path) for name, path in SOURCES.items()
        },
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
