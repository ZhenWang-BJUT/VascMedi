"""Evaluate a completed training run on a separate patient partition."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ._learning import build_reference_bank
from ._prediction import evaluate
from ._runtime import (
    CHECKPOINT, config_namespace, encoder_identity, feature_manifest, file_hash,
    input_paths, make_dataset, make_loader, make_model, make_tokenizer, read_json, write_json,
)


def load_for_test(run, *, train_directory, test_directory, test_features):
    run = Path(run)
    manifest = read_json(run / "training_run.json")
    if file_hash(run / f"{CHECKPOINT}.pth") != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the completed training run")
    if [file_hash(path) for path in input_paths(train_directory)] != manifest["training_input_sha256"]:
        raise ValueError("Training inputs changed")
    cfg = config_namespace(manifest["config"])
    test_features = Path(test_features).resolve()
    for cache in (cfg.visual_features_dir, test_features):
        if encoder_identity(feature_manifest(cache)) != manifest["encoder_sha256"]:
            raise ValueError("Frozen CLIP encoder differs from the completed training run")
    torch.manual_seed(cfg.model_seed)
    tokenizer = make_tokenizer(cfg)
    training = make_dataset(train_directory, tokenizer, cfg, "train")
    testing = make_dataset(test_directory, tokenizer, cfg, "test", training)
    if training.sample_ids != manifest["train_ids"]:
        raise ValueError("Training patients differ from the completed run")
    if set(testing.sample_ids) & set(manifest["validation_ids"]):
        raise ValueError("Patient overlap between validation and test")
    for name in ("mean", "std"):
        np.testing.assert_array_equal(getattr(training, name), manifest["preprocessing"][name])
    ids = manifest["reference_ids"]
    if len(ids) != cfg.reference_count or len(set(ids)) != len(ids) or not set(ids).issubset(training.sample_ids):
        raise ValueError("Invalid training reference bank")

    values = manifest["class_weights"]
    weights = None if values is None else torch.tensor(values, dtype=torch.float32)
    model, device = make_model(cfg, tokenizer, weights)
    model.load_pretrained(run, model_name=CHECKPOINT)
    model.eval()
    references = build_reference_bank(model, training, ids, device)
    weight = float(manifest["reference_lambda"])
    model.set_reference_features(references)
    model.set_reference_lambda(weight)
    return SimpleNamespace(model=model, device=device, training=training, testing=testing,
                           references=references, weight=weight, cfg=cfg,
                           test_features=test_features, reference_ids=ids,
                           checkpoint_sha256=manifest["checkpoint_sha256"])


def test(run, *, train_directory, test_directory, test_features, output):
    bundle = load_for_test(run, train_directory=train_directory, test_directory=test_directory,
                           test_features=test_features)
    bundle.model.visual_features_dir = str(bundle.test_features)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metrics, diagnostics = evaluate(
        bundle.model, make_loader(bundle.testing, bundle.cfg), bundle.device,
        epoch=bundle.cfg.num_epochs, split="test", reference_queries=bundle.references,
        reference_lambda=bundle.weight, output_path=output / "predictions.csv",
    )
    write_json(output / "metrics.json", {"split": "test", "threshold": bundle.cfg.threshold,
                                       "checkpoint_sha256": bundle.checkpoint_sha256, **metrics})
    write_json(output / "diagnostics.json", diagnostics)
    return metrics
