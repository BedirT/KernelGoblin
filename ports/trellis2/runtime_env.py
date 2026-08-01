"""Shared runtime bootstrap for TRELLIS.2 MPS commands and tests."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "ports" / "trellis2"
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
SHIMS = PORT / "shims"


def activate() -> None:
    for path in (str(SHIMS), str(UPSTREAM)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "mps")


def configure_backends() -> None:
    activate()
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    dense_config = importlib.import_module("trellis2.modules.attention.config")
    sparse_config.set_conv_backend("mps")
    sparse_config.set_attn_backend("sdpa")
    dense_config.set_backend("sdpa")

    sparse_attention = importlib.import_module(
        "trellis2.modules.sparse.attention.full_attn"
    )
    sparse_attention.sparse_scaled_dot_product_attention = importlib.import_module(
        "trellis2_mps.sparse_attention"
    ).sparse_scaled_dot_product_attention
    attention_package = importlib.import_module("trellis2.modules.sparse.attention")
    attention_package.sparse_scaled_dot_product_attention = (
        sparse_attention.sparse_scaled_dot_product_attention
    )

    windowed = importlib.import_module("trellis2.modules.sparse.attention.windowed_attn")
    replacements = importlib.import_module("trellis2_mps.windowed_attention")
    windowed.calc_window_partition = replacements.calc_window_partition
    windowed.sparse_windowed_scaled_dot_product_self_attention = (
        replacements.sparse_windowed_scaled_dot_product_self_attention
    )
    windowed.sparse_windowed_scaled_dot_product_cross_attention = (
        replacements.sparse_windowed_scaled_dot_product_cross_attention
    )
    attention_package.calc_window_partition = replacements.calc_window_partition
    attention_package.sparse_windowed_scaled_dot_product_self_attention = (
        replacements.sparse_windowed_scaled_dot_product_self_attention
    )
    attention_package.sparse_windowed_scaled_dot_product_cross_attention = (
        replacements.sparse_windowed_scaled_dot_product_cross_attention
    )
    attention_modules = importlib.import_module(
        "trellis2.modules.sparse.attention.modules"
    )
    attention_modules.sparse_scaled_dot_product_attention = (
        sparse_attention.sparse_scaled_dot_product_attention
    )
    attention_modules.sparse_windowed_scaled_dot_product_self_attention = (
        replacements.sparse_windowed_scaled_dot_product_self_attention
    )

    device = importlib.import_module("trellis2_mps.device")
    sparse_basic = importlib.import_module("trellis2.modules.sparse.basic")
    sparse_basic.VarLenTensor.reduce = device.varlen_reduce
    image_features = importlib.import_module("trellis2.modules.image_feature_extractor")
    image_features.DinoV3FeatureExtractor.__init__ = device.dinov3_init
    image_features.DinoV3FeatureExtractor.__call__ = device.dinov3_call

    rembg = importlib.import_module("trellis2.pipelines.rembg")
    rembg.BiRefNet = device.LazyBiRefNet

    mesh_module = importlib.import_module("trellis2.representations.mesh.base")
    mesh_module.Mesh.fill_holes = device.mesh_fill_holes
    mesh_module.Mesh.remove_faces = device.mesh_remove_faces
    mesh_module.Mesh.simplify = device.mesh_simplify

    import torch
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        torch.cuda.empty_cache = torch.mps.empty_cache
