"""The pinned checkpoint every script in this folder runs, and the versions of NeMo and fla pyproject.toml pins."""

import hashlib
import importlib.metadata
import os

import haas_asr  # noqa: F401  lets NeMo build the checkpoint's KDATransducerDecoder, which its allowlist refuses otherwise
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

NEMO = importlib.metadata.version("nemo-toolkit")
FLA = importlib.metadata.version("flash-linear-attention")
MODELS = {
    "Haas-ja-v1.0": {
        "repository": "third-intelligence/Haas-ja-v1.0",
        "revision": "660728d39019e5e065046154beca44ac9a66a540",
        "file": "Haas-ja-v1.0.nemo",
        "size": 1232384000,
        "sha256": "c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa",
    },
}


def checkpoint(pin):
    """The local path of a pinned .nemo checkpoint, downloaded on first use and checked against its pin."""
    path = hf_hub_download(pin["repository"], pin["file"], revision=pin["revision"])
    size = os.path.getsize(path)
    if size != pin["size"]:
        raise SystemExit(f"{path} has {size} bytes, not the pinned {pin['size']}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {pin['sha256']}")
    return path


def restore(pin, device="cpu"):
    """
    The official model of a pinned checkpoint, in float32 and in evaluation, on `device`. The prediction network's
    Kimi Delta Attention runs on fla's Triton kernels, which need CUDA; TF32 is off, so that every matrix product sums in
    float32 as the CPU's would.
    """
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ASRModel.restore_from(checkpoint(pin), map_location=device)
    return model.float().eval()
