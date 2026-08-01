"""Low-disk checkpoint loader for TRELLIS.2 model components."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS = Path(
    os.environ.get(
        "KG_TRELLIS2_DOWNLOAD_DIR", ROOT / "build" / "trellis2" / "downloads"
    )
).expanduser()
REVISIONS = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
}
GENERATED_BUFFER_ALLOWLIST = {"rope_phases"}


def normalize_checkpoint_path(path: str) -> str:
    duplicated = "microsoft/TRELLIS.2-4B/microsoft/"
    if path.startswith(duplicated):
        return path[len("microsoft/TRELLIS.2-4B/"):]
    return path


def streaming_from_pretrained(path: str, **kwargs):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from trellis2 import models

    local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")
    temporary = False
    model_file = None

    try:
        if local:
            config_file = Path(f"{path}.json")
            model_file = Path(f"{path}.safetensors")
        else:
            parts = path.split("/")
            if len(parts) < 3:
                raise ValueError(
                    f"checkpoint path must include a repository and model: {path}"
                )
            repo_id = "/".join(parts[:2])
            model_name = "/".join(parts[2:])
            DOWNLOADS.mkdir(parents=True, exist_ok=True)
            revision = REVISIONS[repo_id]
            config_file = Path(
                hf_hub_download(repo_id, f"{model_name}.json", revision=revision)
            )
            print(f"Downloading {repo_id}/{model_name}.safetensors", flush=True)
            temporary = True
            model_file = Path(
                hf_hub_download(
                    repo_id,
                    f"{model_name}.safetensors",
                    revision=revision,
                    local_dir=DOWNLOADS,
                )
            )

        with config_file.open() as stream:
            config = json.load(stream)
        model = getattr(models, config["name"])(**config["args"], **kwargs)
        state = load_file(model_file)
        incompatible = model.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys) - GENERATED_BUFFER_ALLOWLIST
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch for {path}: "
                f"missing={sorted(missing)[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        del state
    finally:
        if temporary:
            if model_file is not None:
                model_file.unlink(missing_ok=True)
            shutil.rmtree(DOWNLOADS / ".cache", ignore_errors=True)
            gc.collect()
    return model


def lazy_from_pretrained(path: str, **kwargs):
    import torch.nn as nn

    class LazyModel(nn.Module):
        def __init__(self, checkpoint_path, checkpoint_kwargs):
            super().__init__()
            object.__setattr__(self, "_checkpoint_path", checkpoint_path)
            object.__setattr__(self, "_checkpoint_kwargs", checkpoint_kwargs)

        def _load(self):
            if "loaded" not in self._modules:
                model = streaming_from_pretrained(
                    self._checkpoint_path, **self._checkpoint_kwargs
                ).eval()
                # add_module() checks hasattr(), which invokes this proxy's
                # __getattr__ and recursively tries to load the same model.
                self._modules["loaded"] = model
            return self._modules["loaded"]

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._load(), name)

        def forward(self, *args, **forward_kwargs):
            return self._load()(*args, **forward_kwargs)

        def to(self, *args, **to_kwargs):
            self._load().to(*args, **to_kwargs)
            return self

        def cpu(self):
            if "loaded" in self._modules:
                self._modules["loaded"].cpu()
            return self

        def unload(self):
            if "loaded" in self._modules:
                model = self._modules.pop("loaded")
                model.cpu()
                del model
            return self

        def eval(self):
            self.training = False
            if "loaded" in self._modules:
                self._modules["loaded"].eval()
            return self

    return LazyModel(normalize_checkpoint_path(path), kwargs)


def install(*, lazy: bool = False) -> None:
    import huggingface_hub
    from trellis2 import models

    models.from_pretrained = lazy_from_pretrained if lazy else streaming_from_pretrained
    original_download = huggingface_hub.hf_hub_download

    def pinned_download(repo_id, filename, *args, **kwargs):
        if repo_id in REVISIONS and kwargs.get("revision") is None:
            kwargs["revision"] = REVISIONS[repo_id]
        return original_download(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = pinned_download
