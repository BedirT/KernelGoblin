#!/usr/bin/env python3
"""Capture block 0 from the pinned dense sparse-structure flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
sys.path.insert(0, str(PORT / "shims"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PORT))

import runtime_env

runtime_env.activate()

SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
WEIGHT_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
WEIGHT_SHA256 = "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic(count: int, multiplier: float, scale: float) -> torch.Tensor:
    return torch.tensor(
        [math.sin(index * multiplier) * scale for index in range(count)],
        dtype=torch.float32,
    ).to(torch.bfloat16)


def require_pinned_source() -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise SystemExit(f"upstream checkout is {revision}, expected {SOURCE_REVISION}")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM
    ).returncode != 0:
        raise SystemExit("upstream checkout has tracked modifications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned sparse flow")

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.modules.attention import RotaryPositionEmbedder
    from trellis2.modules.transformer import ModulatedTransformerCrossBlock
    from trellis2.modules.utils import convert_module_to

    runtime_env.configure_backends()
    block = ModulatedTransformerCrossBlock(
        1536,
        1024,
        num_heads=12,
        mlp_ratio=5.3334,
        attn_mode="full",
        share_mod=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        use_rope=True,
    ).eval()
    block.apply(partial(convert_module_to, dtype=torch.bfloat16))
    state = {}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if name.startswith("blocks.0."):
                state[name.removeprefix("blocks.0.")] = checkpoint.get_tensor(name)
    block.load_state_dict(state, strict=True)

    tokens, context_tokens = 2, 2
    features = deterministic(tokens * 1536, 0.013, 0.35).reshape(1, tokens, 1536)
    modulation = deterministic(9216, 0.007, 0.2).reshape(1, 9216)
    context = deterministic(context_tokens * 1024, 0.015, 0.25).reshape(
        1, context_tokens, 1024
    )
    coordinates = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32)
    phases = RotaryPositionEmbedder(128, 3)(coordinates)
    with torch.inference_mode():
        chunks = (block.modulation + modulation).type(modulation.dtype).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        stages = {}
        x = features
        h = block.norm1(x)
        stages["norm1"] = h
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        stages["self_input"] = h
        h = block.self_attn(h, phases=phases)
        stages["self_output"] = h
        x = x + h * gate_msa.unsqueeze(1)
        stages["after_self"] = x
        h = block.norm2(x)
        stages["norm2"] = h
        h = block.cross_attn(h, context)
        stages["cross_output"] = h
        x = x + h
        stages["after_cross"] = x
        h = block.norm3(x)
        stages["norm3"] = h
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        stages["mlp_input"] = h
        h = block.mlp.mlp[0](h)
        stages["mlp_hidden_linear"] = h
        h = block.mlp.mlp[1](h)
        stages["mlp_hidden_gelu"] = h
        h = block.mlp.mlp[2](h)
        stages["mlp_output"] = h
        output = x + h * gate_mlp.unsqueeze(1)
        stages["output"] = output

    output_bits = output.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bits)
    trace_path = args.output.with_suffix(args.output.suffix + ".trace")
    trace_offsets = {}
    trace_payload = bytearray()
    for name, tensor in stages.items():
        bits = tensor.contiguous().view(torch.uint16).cpu().numpy().tobytes()
        trace_offsets[name] = [len(trace_payload), len(trace_payload) + len(bits)]
        trace_payload.extend(bits)
    trace_path.write_bytes(trace_payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 sparse-structure block BF16 oracle v1",
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "source_model": SparseStructureFlowModel.__name__,
        "block": 0,
        "batch_count": 1,
        "tokens": tokens,
        "coordinates": [[0, 0, 0, 0], [0, 0, 0, 1]],
        "context_tokens": context_tokens,
        "channels": 1536,
        "context_channels": 1024,
        "use_rope": True,
        "input_formula": "bf16(sin(i * 0.013) * 0.35)",
        "modulation_formula": "bf16(sin(i * 0.007) * 0.2)",
        "context_formula": "bf16(sin(i * 0.015) * 0.25)",
        "output_bytes": len(output_bits),
        "output_sha256": hashlib.sha256(output_bits).hexdigest(),
        "trace_file": trace_path.name,
        "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "trace_offsets": trace_offsets,
        "oracle": "pinned Torch CPU ModulatedTransformerCrossBlock",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
