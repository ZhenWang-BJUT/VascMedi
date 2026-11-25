"""Measure prediction changes after occluding retinal patches."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .._runtime import encoder_identity, feature_manifest
from ..datasets.visual import encoder_fingerprint, load_clip, read_images
from ._retina import _clip_encode, _fundus_mask, eligible_cells


@torch.inference_mode()
def analyze_patches(bundle, image_manifest, clip_model, output):
    cache = feature_manifest(bundle.test_features)
    if encoder_fingerprint(clip_model) != encoder_identity(cache):
        raise ValueError("Use the same local CLIP model to export and occlude the test images")
    entries = read_images(image_manifest)
    lookup = {patient: index for index, patient in enumerate(bundle.testing.sample_ids)}
    if any(patient not in lookup for patient, _ in entries):
        raise ValueError("Patch analysis accepts test patients only")

    processor, encoder = load_clip(clip_model, bundle.device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle.model.visual_features_dir = str(bundle.test_features)
    batch_size = cache.get("batch_size", 8)
    for (patient, view), path in sorted(entries.items()):
        item = bundle.testing[lookup[patient]]
        sample = item["sample_id"][None]
        device = bundle.device
        model = bundle.model
        original_local, original_global, valid_views = model._get_image_features_by_id(sample)
        if not bool(valid_views[0, view]):
            raise ValueError("The requested patient view is missing from the test feature cache")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        field = _fundus_mask(image)
        base = np.asarray(image)
        height, width = field.shape
        patches = eligible_cells(image, field)
        color = np.median(base[field], axis=0).astype(np.uint8)

        def occlude(cells):
            array = base.copy()
            for row, col in cells:
                y = slice(round(row * height / 16), round((row + 1) * height / 16))
                x = slice(round(col * width / 16), round((col + 1) * width / 16))
                array[y, x][field[y, x]] = color
            return Image.fromarray(array)

        def score(images, check_clean=False):
            local, global_features = _clip_encode(images, processor, encoder, device)
            scores = []
            for loc, glob in zip(local, global_features):
                if check_clean:
                    torch.testing.assert_close(loc.to(original_local), original_local[0, view],
                                               atol=1e-3, rtol=1e-3)
                    torch.testing.assert_close(glob.to(original_global), original_global[0, view],
                                               atol=1e-3, rtol=1e-3)
                features = original_local.clone()
                pooled = original_global.clone()
                features[0, view] = loc.to(features)
                pooled[0, view] = glob.to(pooled)
                logits = model(
                    sample_id=sample,
                    tabular_features=item["tabular_features"][None].to(device),
                    tabular_observed_mask=item["tabular_observed_mask"][None].to(device),
                    input_ids=item["input_ids"][None].to(device),
                    attention_mask=item["attention_mask"][None].to(device),
                    reference_queries=bundle.references, reference_lambda=bundle.weight,
                    image_features_override=(features, pooled, valid_views),
                )["logits"][0].float()
                scores.append((float(logits.softmax(-1)[1]), float(logits[1] - logits[0])))
            return np.array(scores)

        clean = score([image], check_clean=True)[0]
        maps = np.full((16, 16, 2), np.nan, dtype=np.float32)
        for start in range(0, len(patches), batch_size):
            batch = patches[start:start + batch_size]
            for cell, delta in zip(batch, clean - score([occlude([cell]) for cell in batch])):
                maps[cell] = delta
        name = output / f"{patient}_{view}"
        np.savez_compressed(
            name.with_suffix(".npz"), delta_probability=maps[:, :, 0],
            delta_margin=maps[:, :, 1], field=field, image=base,
        )

        direction = 1 if int(item["labels"]) else -1
        rng = np.random.default_rng(bundle.cfg.seed + patient * 4 + view)
        records = []
        for index, metric in enumerate(("probability", "margin")):
            ranked = sorted(patches, key=lambda cell: direction * maps[cell][index], reverse=True)
            for fraction in (0.05, 0.10, 0.20):
                count = max(1, round(len(patches) * fraction))
                groups = [ranked[:count]]
                groups += [[patches[i] for i in rng.choice(len(patches), count, replace=False)]
                           for _ in range(20)]
                for start in range(0, len(groups), batch_size):
                    batch = groups[start:start + batch_size]
                    for offset, response in enumerate(clean - score([occlude(cells) for cells in batch])):
                        draw = start + offset
                        records.append((metric, fraction, "top" if draw == 0 else f"random_{draw}",
                                        *(direction * response)))
        with name.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("ranking_metric", "fraction", "condition",
                             "label_aligned_delta_probability", "label_aligned_delta_margin"))
            writer.writerows(records)


def plot_patch(path, output, comparison=None):
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as data:
        image, field, delta = data["image"], data["field"], data["delta_probability"]
    maps = [("Patch response", delta)]
    if comparison is not None:
        with np.load(comparison, allow_pickle=False) as data:
            if not np.array_equal(data["image"], image) or not np.array_equal(data["field"], field):
                raise ValueError("Compare maps for the same image")
            maps.insert(0, ("Comparison response", data["delta_probability"]))
    scale = max(1e-12, *(float(np.nanmax(np.abs(values))) for _, values in maps))
    figure, axes = plt.subplots(1, len(maps) + 1, figsize=((len(maps) + 1) * 3, 3.4))
    axes[0].imshow(image)
    axes[0].set_title("Fundus image")
    height, width = field.shape
    rows = np.minimum(np.arange(height) * 16 // height, 15)
    cols = np.minimum(np.arange(width) * 16 // width, 15)
    for axis, (label, values) in zip(axes[1:], maps):
        axis.imshow(image)
        layer = np.where(field, values[np.ix_(rows, cols)], np.nan)
        heatmap = axis.imshow(layer, alpha=0.55, cmap="coolwarm", vmin=-scale, vmax=scale)
        axis.set_title(label)
    for axis in axes:
        axis.set_axis_off()
    figure.colorbar(heatmap, ax=list(axes), orientation="horizontal", pad=0.03,
                    label="Positive-class probability change")
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
