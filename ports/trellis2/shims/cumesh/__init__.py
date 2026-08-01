"""CPU fallback for the CuMesh methods used by TRELLIS.2 inference output."""

from __future__ import annotations

import numpy as np
import torch
import trimesh


class CuMesh:
    def __init__(self):
        self.mesh = None

    def init(self, vertices, faces):
        self.device = vertices.device
        self.mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        )

    @property
    def num_boundaries(self):
        edges = self.mesh.edges_sorted
        return int(np.any(np.unique(edges, axis=0, return_counts=True)[1] == 1))

    @property
    def num_boundary_loops(self):
        return self.num_boundaries

    def __getattr__(self, name):
        if name.startswith("get_") or name.startswith("read_manifold_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def fill_holes(self, max_hole_perimeter=3e-2):
        trimesh.repair.fill_holes(self.mesh)

    def simplify(self, target, verbose=False, options=None):
        if len(self.mesh.faces) > target:
            try:
                self.mesh = self.mesh.simplify_quadric_decimation(face_count=target)
            except BaseException:
                pass

    def read(self):
        vertices = torch.as_tensor(np.asarray(self.mesh.vertices).copy(), dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(np.asarray(self.mesh.faces).copy(), dtype=torch.int32, device=self.device)
        return vertices, faces
