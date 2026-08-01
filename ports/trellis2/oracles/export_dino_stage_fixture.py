#!/usr/bin/env python3
"""Export a complete tiny DINOv3 conditioning trace from pinned upstream code."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import DINOv3ViTModel
from transformers.models.dinov3_vit import modeling_dinov3_vit


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DINO_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
DINO_CONFIG_SHA256 = "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
TRANSFORMERS_VERSION = "5.3.0"
TRANSFORMERS_DINO_SOURCE_SHA256 = (
    "f1e8fdfe586f6919e229b72b9538a088af500a67c65901c88ca85ca7956dc0bd"
)
SOURCE_PATH = "trellis2/modules/image_feature_extractor.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    path.write_bytes(values.tobytes())
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "Tests" / "KernelGoblinTrellis2Tests" / "Fixtures" / "dino-stage-tiny",
    )
    args = parser.parse_args()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != SOURCE_REVISION:
        raise SystemExit(f"upstream revision mismatch: {head}")
    checkpoint = args.checkpoint.expanduser().absolute()
    if checkpoint.parent.name != DINO_REVISION:
        raise SystemExit(
            "DINO checkpoint must come from the exact pinned snapshot directory"
        )
    if sha256(checkpoint) != DINO_SHA256:
        raise SystemExit("DINO checkpoint SHA-256 mismatch")
    config = checkpoint.parent / "config.json"
    if not config.is_file() or sha256(config) != DINO_CONFIG_SHA256:
        raise SystemExit("DINO config SHA-256 mismatch")
    transformers_source = Path(inspect.getfile(modeling_dinov3_vit)).resolve()
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            f"Transformers version mismatch: {transformers.__version__}"
        )
    if sha256(transformers_source) != TRANSFORMERS_DINO_SOURCE_SHA256:
        raise SystemExit("Transformers DINO source SHA-256 mismatch")

    torch.set_grad_enabled(False)
    model = DINOv3ViTModel.from_pretrained(
        checkpoint.parent,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().cpu()
    image_size = args.image_size
    if image_size <= 0 or image_size % 16 != 0:
        raise SystemExit("image size must be a positive multiple of 16")
    pixel_count = 3 * image_size * image_size
    pixels = torch.arange(pixel_count, dtype=torch.float32).reshape(1, 3, image_size, image_size)
    pixels = torch.sin(pixels * 0.013) * 0.75 + torch.cos(pixels * 0.007) * 0.25

    stages: list[tuple[str, torch.Tensor]] = [("normalized_pixels", pixels)]
    hidden = model.embeddings(pixels, bool_masked_pos=None)
    if image_size <= 64:
        stages.append(("embeddings", hidden))
    positions = model.rope_embeddings(pixels)
    for index, layer in enumerate(model.layer):
        hidden = layer(hidden, position_embeddings=positions)
        if image_size <= 64:
            stages.append((f"block_{index}", hidden))
    output = F.layer_norm(hidden, hidden.shape[-1:])
    if image_size <= 64:
        stages.append(("final_parameter_free_layer_norm", output))

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = prefix.with_suffix(".f32")
    trace_path = prefix.with_suffix(".f32.trace")
    metadata_path = prefix.with_suffix(".f32.json")
    output_sha = write_f32(output_path, output)

    trace_bytes = bytearray()
    trace_stages = []
    float_offset = 0
    for name, tensor in stages:
        values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        payload = values.tobytes()
        trace_bytes.extend(payload)
        trace_stages.append({
            "name": name,
            "shape": list(values.shape),
            "float_offset": float_offset,
            "float_count": int(values.size),
        })
        float_offset += int(values.size)
    trace_path.write_bytes(trace_bytes)
    trace_sha = sha256(trace_path)

    metadata = {
        "format": "KernelGoblin TRELLIS.2 complete DINOv3 F32 oracle v1",
        "upstream_repository": "https://github.com/microsoft/TRELLIS.2",
        "upstream_revision": SOURCE_REVISION,
        "upstream_source": SOURCE_PATH,
        "dino_repository": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dino_revision": DINO_REVISION,
        "dino_config_sha256": DINO_CONFIG_SHA256,
        "checkpoint_sha256": DINO_SHA256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_dino_source_sha256": TRANSFORMERS_DINO_SOURCE_SHA256,
        "attention_implementation": "eager",
        "image_layout": "NCHW",
        "image_size": image_size,
        "patch_size": 16,
        "prefix_tokens": 5,
        "token_count": 5 + (image_size // 16) ** 2,
        "hidden_size": 1024,
        "layers": 24,
        "heads": 16,
        "head_dimension": 64,
        "mlp_hidden_size": 4096,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 100.0,
        "final_norm": "parameter-free torch.nn.functional.layer_norm",
        "output_file": output_path.name,
        "output_sha256": output_sha,
        "trace_file": trace_path.name,
        "trace_sha256": trace_sha,
        "trace_stages": trace_stages,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output_path} sha256={output_sha}")
    print(f"wrote {trace_path} sha256={trace_sha}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
