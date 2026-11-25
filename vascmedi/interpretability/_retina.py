"""Retinal field and frozen encoder helpers."""
from typing import Any, Sequence, Tuple

from PIL import Image
import torch

def _fundus_mask(image):
    """Largest nonblack retinal field, hole-filled; scanner text is excluded."""
    import numpy as np
    from scipy import ndimage
    gray = np.asarray(image.convert('L'))
    seed = ndimage.binary_opening(gray > 12, structure=np.ones((5, 5), dtype=bool))
    seed = ndimage.binary_closing(seed, structure=np.ones((5, 5), dtype=bool))
    labels, count = ndimage.label(seed)
    if not count:
        raise ValueError('No retinal field')
    size = np.bincount(labels.ravel())
    size[0] = 0
    return ndimage.binary_fill_holes(labels == int(size.argmax()))


def eligible_cells(image, fundus):
    cells = []
    width, height = image.size
    for r in range(16):
        for c in range(16):
            if fundus[round(r*height/16):round((r+1)*height/16),
                      round(c*width/16):round((c+1)*width/16)].any():
                cells.append((r, c))
    if not cells:
        raise ValueError('No retinal cells')
    return cells


def _clip_encode(
    images: Sequence[Image.Image], processor: Any, vision_model: Any, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(images=list(images), return_tensors="pt", padding=True).to(device)
    output = vision_model(**inputs)
    local = output.last_hidden_state.detach().cpu()
    global_ = output.pooler_output.detach().cpu()
    if local.ndim != 3 or global_.ndim != 2 or local.shape[1:] != (257, 1024) or global_.shape[1] != 1024:
        raise RuntimeError(f"Unexpected CLIP features: local={tuple(local.shape)}, global={tuple(global_.shape)}")
    return local, global_
