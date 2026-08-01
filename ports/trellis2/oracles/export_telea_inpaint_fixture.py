#!/usr/bin/env python3
"""Export byte-exact OpenCV 4.13 Telea fixtures for the native PBR baker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


OPENCV_REVISION = "fe38fc608f6acb8b68953438a62305d8318f4fcd"
SOURCE_PATH = "modules/photo/src/inpaint.cpp"
SOURCE_SHA256 = "15c66e1f742fd8a08590996e80e06895c5355126bd9457ab613be03179019734"


def fixture_cases():
    scalar = ((np.arange(20, dtype=np.uint16) * 37 + 11) % 256).astype(np.uint8).reshape(4, 5)
    yield "empty-mask-scalar", scalar, np.zeros((4, 5), np.uint8), 1
    yield "full-mask-scalar", scalar, np.full((4, 5), 255, np.uint8), 1

    scalar = ((np.arange(49, dtype=np.uint16) * 29 + 7) % 256).astype(np.uint8).reshape(7, 7)
    mask = np.zeros((7, 7), np.uint8)
    mask[2:5, 3] = 1
    mask[3, 2:5] = 1
    yield "symmetric-cross-scalar", scalar, mask, 1

    scalar = ((np.arange(48, dtype=np.uint16) * 53 + 19) % 256).astype(np.uint8).reshape(6, 8)
    mask = np.zeros((6, 8), np.uint8)
    mask[0:2, 0:3] = 255
    mask[3, 6] = 7
    mask[5, 4:6] = 1
    yield "border-disconnected-scalar", scalar, mask, 1

    indices = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)
    rgb = np.stack([
        (indices * 17 + 3) % 256,
        (indices * 31 + 91) % 256,
        (indices * 47 + 151) % 256,
    ], axis=-1).astype(np.uint8)
    mask = np.zeros((7, 9), np.uint8)
    mask[2:5, 2:7] = 1
    mask[0, 8] = 255
    mask[6, 0:2] = 1
    yield "rgb-radius-three", rgb, mask, 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if cv2.__version__ != "4.13.0":
        raise SystemExit("Telea fixture export requires OpenCV 4.13.0")
    payload = bytearray()
    layouts = []
    for name, source, mask, radius in fixture_cases():
        expected = cv2.inpaint(source, mask, radius, cv2.INPAINT_TELEA)
        case = {
            "name": name,
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
            "channels": 1 if source.ndim == 2 else int(source.shape[2]),
            "radius": radius,
        }
        for field, value in (("source", source), ("mask", mask), ("expected", expected)):
            data = np.ascontiguousarray(value).tobytes()
            case[field] = {"offset": len(payload), "length": len(data)}
            payload.extend(data)
        layouts.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin OpenCV Telea uint8 oracle v1",
        "opencv_version": cv2.__version__,
        "opencv_revision": OPENCV_REVISION,
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "license": "Apache-2.0",
        "cases": layouts,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
