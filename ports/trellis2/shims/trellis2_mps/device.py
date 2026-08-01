"""Device-neutral replacements for upstream helpers that hard-code CUDA."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"


def varlen_reduce(self, op: str, dim=None, keepdim: bool = False):
    if isinstance(dim, int):
        dim = (dim,)
    reduction = getattr(self.feats, op)
    reduced = reduction(dim=dim, keepdim=keepdim)
    if dim is None or 0 in dim:
        return reduced

    outputs = []
    for segment in self.layout:
        # The implicit variable-length axis is replaced by the batch axis, just
        # like torch.segment_reduce; it does not gain another keepdim axis.
        outputs.append(getattr(reduced[segment], op)(dim=0))
    return torch.stack(outputs, dim=0)


def dinov3_init(self, model_name: str, image_size=512):
    from torchvision import transforms
    from transformers import DINOv3ViTModel

    self.model_name = model_name
    self.model = DINOv3ViTModel.from_pretrained(
        model_name, revision=DINO_REVISION
    ).eval()
    self.image_size = image_size
    self.transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


@torch.no_grad()
def dinov3_call(self, image):
    device = next(self.model.parameters()).device
    if isinstance(image, torch.Tensor):
        assert image.ndim == 4, "image tensor should be batched (B, C, H, W)"
    elif isinstance(image, list):
        assert all(isinstance(item, Image.Image) for item in image)
        image = [item.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS) for item in image]
        image = [np.asarray(item.convert("RGB"), dtype=np.float32) / 255 for item in image]
        image = torch.stack([torch.from_numpy(item).permute(2, 0, 1) for item in image])
    else:
        raise ValueError(f"unsupported image type: {type(image)}")
    image = self.transform(image).to(device)
    return self.extract_features(image)


class LazyBiRefNet:
    """Delay the optional background-removal checkpoint until it is used."""

    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cpu")

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForImageSegmentation

            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name, revision=RMBG_REVISION, trust_remote_code=True
            ).eval().to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("mps")

    def __call__(self, image: Image.Image) -> Image.Image:
        from torchvision import transforms

        self._load()
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        original_size = image.size
        inputs = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(inputs)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(prediction).resize(original_size)
        result = image.copy()
        result.putalpha(mask)
        return result


def mesh_fill_holes(self, max_hole_perimeter=3e-2):
    import cumesh

    mesh = cumesh.CuMesh()
    mesh.init(self.vertices, self.faces)
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    self.vertices, self.faces = mesh.read()


def mesh_remove_faces(self, face_mask):
    keep = ~face_mask.to(torch.bool)
    self.faces = self.faces[keep]


def mesh_simplify(self, target=1_000_000, verbose=False, options=None):
    import cumesh

    mesh = cumesh.CuMesh()
    mesh.init(self.vertices, self.faces)
    mesh.simplify(target, verbose=verbose, options=options or {})
    self.vertices, self.faces = mesh.read()
