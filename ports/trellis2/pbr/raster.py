"""Binary bridge to the physical Metal UV raster kernel."""

from __future__ import annotations

import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import validate_mesh


ROOT = Path(__file__).resolve().parents[3]
EXECUTABLE = (
    ROOT / "build" / "trellis2-uv-raster" / "kernels" / "trellis2" /
    "uv_raster" / "kg_trellis2_uv_raster_cli"
)
_HEADER = struct.Struct("<4sIIII")


@dataclass(frozen=True)
class RasterResult:
    positions: np.ndarray
    face_ids: np.ndarray
    backend: str


def rasterize_metal(vertices, faces, uvs, *, width: int, height: int | None = None) -> RasterResult:
    positions, triangles = validate_mesh(vertices, faces)
    texcoords = np.ascontiguousarray(uvs, dtype=np.float32)
    height = width if height is None else height
    if texcoords.shape != (len(positions), 2):
        raise ValueError("uvs must have shape [V, 2]")
    if not np.isfinite(texcoords).all():
        raise ValueError("uvs must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("raster dimensions must be positive")
    if not EXECUTABLE.is_file():
        raise RuntimeError(
            "Metal UV raster is not built; run `./kg setup trellis2/uv_raster`"
        )

    with tempfile.TemporaryDirectory(prefix="kg-uv-raster-") as directory:
        input_path = Path(directory) / "input.bin"
        output_path = Path(directory) / "output.bin"
        with input_path.open("wb") as stream:
            stream.write(_HEADER.pack(b"KGUV", width, height, len(positions), len(triangles)))
            stream.write(positions.tobytes())
            stream.write(texcoords.tobytes())
            stream.write(triangles.tobytes())
        completed = subprocess.run(
            [str(EXECUTABLE), str(input_path), str(output_path)],
            check=True, capture_output=True, text=True,
        )
        with output_path.open("rb") as stream:
            header = stream.read(_HEADER.size)
            magic, output_width, output_height, _, _ = _HEADER.unpack(header)
            if magic != b"KGUR" or (output_width, output_height) != (width, height):
                raise RuntimeError("Metal UV raster returned an invalid header")
            pixel_count = width * height
            position_bytes = stream.read(pixel_count * 4 * 4)
            face_bytes = stream.read(pixel_count * 4)
            if len(position_bytes) != pixel_count * 16 or len(face_bytes) != pixel_count * 4:
                raise RuntimeError("Metal UV raster returned truncated output")
            if stream.read(1):
                raise RuntimeError("Metal UV raster returned trailing output")
        return RasterResult(
            positions=np.frombuffer(position_bytes, dtype="<f4").reshape(height, width, 4).copy(),
            face_ids=np.frombuffer(face_bytes, dtype="<u4").reshape(height, width).copy(),
            backend=completed.stdout.strip(),
        )
