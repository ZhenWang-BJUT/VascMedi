"""Fit a model using the training and validation partitions."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from ._learning import build_reference_bank, lr_factor, select_reference_ids, train_one_epoch
from ._prediction import evaluate
from ._runtime import (
    CHECKPOINT, config_namespace, encoder_identity, feature_manifest, file_hash,
    input_paths, make_dataset, make_loader, make_model, make_tokenizer, read_json, write_json,
)


def _optimizer(model, cfg):
    groups = {"lora": [], "auxiliary": [], "scalars": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in {"reference_lambda_logit", "front_door_attn.query_residual_logit"}:
            groups["scalars"].append(parameter)
        elif "lora_" in name.lower():
            groups["lora"].append(parameter)
        else:
            groups["auxiliary"].append(parameter)
    if not groups["lora"] or not groups["auxiliary"]:
        raise ValueError("Expected trainable LoRA and auxiliary parameters")
    options = [
        {"params": groups["lora"], "lr": cfg.lora_lr},
        {"params": groups["auxiliary"], "lr": cfg.lr},
    ]
    if groups["scalars"]:
        options.append({"params": groups["scalars"], "lr": cfg.scalar_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(options, weight_decay=cfg.weight_decay)


def train(config_path, *, train_directory, validation_directory, features, model_path, output):
    config = read_json(config_path)
    config.update(model_name=str(Path(model_path).resolve()),
                  visual_features_dir=str(Path(features).resolve()))
    cfg = config_namespace(config)
    encoder_hash = encoder_identity(feature_manifest(cfg.visual_features_dir))
    torch.manual_seed(cfg.model_seed)
    random.seed(cfg.model_seed)
    np.random.seed(cfg.model_seed)

    tokenizer = make_tokenizer(cfg)
    training = make_dataset(train_directory, tokenizer, cfg, "train")
    validation = make_dataset(validation_directory, tokenizer, cfg, "validation", training)
    train_hashes = [file_hash(path) for path in input_paths(train_directory)]
    reference_ids = select_reference_ids(training, cfg.reference_count, cfg.reference_seed)
    generator = torch.Generator().manual_seed(cfg.model_seed)
    sampler = None
    weights = training.class_weights.clone()
    if cfg.use_weighted_sampler:
        sample_weights = weights[torch.tensor(training.sample_labels)].double()
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True,
            generator=torch.Generator().manual_seed(cfg.model_seed),
        )
        weights = None
    else:
        weights[0] *= cfg.negative_class_weight_scale
        weights[1] *= cfg.positive_class_weight_scale
    train_loader = make_loader(training, cfg, train=True, sampler=sampler, generator=generator)
    validation_loader = make_loader(validation, cfg)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, device = make_model(cfg, tokenizer, weights)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    history = []
    training_references = None
    for epoch in range(1, cfg.num_epochs + 1):
        if training_references is None or (epoch - 1) % cfg.reference_refresh_epochs == 0:
            training_references = build_reference_bank(model, training, reference_ids, device)
        model.set_reference_features(training_references[:cfg.reference_count])
        model.set_reference_lambda(cfg.reference_lambda)
        model.lambda_learning_active = epoch > cfg.lambda_warmup_epochs
        train_one_epoch(
            model, train_loader, optimizer, device, epoch, reference_ids,
            training_references, cfg.reference_count,
            None if model.learn_reference_lambda else cfg.reference_lambda,
        )
        references = build_reference_bank(model, training, reference_ids[:cfg.reference_count], device)
        fusion_weight = float(model.effective_reference_lambda())
        metrics, _ = evaluate(
            model, validation_loader, device, epoch=epoch, split="validation",
            reference_queries=references, reference_lambda=fusion_weight,
        )
        history.append({"epoch": epoch, **metrics})
        write_json(output / "validation_metrics.json", history)
        scheduler.step()

    model.save_pretrained(output, model_name=CHECKPOINT)
    write_json(output / "training_run.json", {
        "config": config,
        "checkpoint_sha256": file_hash(output / f"{CHECKPOINT}.pth"),
        "encoder_sha256": encoder_hash,
        "training_input_sha256": train_hashes,
        "train_ids": training.sample_ids,
        "validation_ids": validation.sample_ids,
        "reference_ids": reference_ids[:cfg.reference_count],
        "spare_reference_id": reference_ids[cfg.reference_count],
        "reference_lambda": fusion_weight,
        "class_weights": weights.tolist() if weights is not None else None,
        "preprocessing": {"mean": training.mean.tolist(), "std": training.std.tolist()},
    })
    logging.getLogger(__name__).info("Training complete: %s", output)
