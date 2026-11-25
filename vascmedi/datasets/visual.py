"""Extract label-independent frozen CLIP features."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from .._runtime import file_hash, write_json
from ..interpretability._retina import _clip_encode


def encoder_fingerprint(model_directory):
    files = sorted(path for path in Path(model_directory).iterdir()
                   if path.is_file() and path.suffix in {".json", ".bin", ".safetensors"})
    if not files:
        raise ValueError("Expected a local CLIP model with processor settings and weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.name}:{file_hash(path)}\n".encode())
    return digest.hexdigest()


def load_clip(model_directory, device):
    processor = CLIPImageProcessor.from_pretrained(model_directory, local_files_only=True)
    model = CLIPVisionModel.from_pretrained(model_directory, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    return processor, model


def read_images(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        entries = {}
        for row in csv.DictReader(handle):
            key = (int(row["sample_id"]), int(row["v_idx"]))
            if key in entries or not 0 <= key[1] < 4:
                raise ValueError("Expected unique patient/view pairs with v_idx 0–3")
            entries[key] = (manifest.parent / row["image_path"]).resolve()
    if not entries:
        raise ValueError("Empty image manifest")
    return entries


@torch.inference_mode()
def export_features(manifest, model_directory, output, batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    entries = sorted(read_images(manifest).items())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprint = encoder_fingerprint(model_directory)
    processor, vision = load_clip(model_directory, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("pooler_output", "last_hidden_state"):
        (output / name).mkdir()
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        images = []
        for _, path in batch:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        local, global_features = _clip_encode(images, processor, vision, device)
        for index, ((patient, view), _) in enumerate(batch):
            name = f"{patient}_{view}.npz"
            np.savez_compressed(output / "pooler_output" / name,
                                pooler_output=global_features[index:index + 1].numpy())
            np.savez_compressed(output / "last_hidden_state" / name,
                                last_hidden_state=local[index:index + 1].numpy())
    write_json(output / "feature_provenance.json", {
        "encoder_sha256": fingerprint,
        "image_manifest_sha256": file_hash(manifest),
        "feature_shape": [257, 1024],
        "encoder_frozen": True,
        "task_supervised_training": False,
        "training_steps": 0,
        "export_complete": True,
        "batch_size": batch_size,
    })
