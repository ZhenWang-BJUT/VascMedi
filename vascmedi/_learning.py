"""Training primitives shared by AIS and CHD."""
import logging
import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .datasets.retinal import collate_fn

logger = logging.getLogger(__name__)

def lr_factor(index):
    return .1 + .9 * (1 + math.cos(math.pi * min(10, max(0, index - 1)) / 10)) / 2


def select_reference_ids(train_dataset, count, seed):
    """Uniform train-patient sampling; one spare donor supports leave-one-out training."""
    ids = np.asarray(train_dataset.sample_ids, dtype=np.int64)
    count = int(count)
    if count < 1 or len(ids) < count + 1:
        raise ValueError(f"reference_count={count} needs at least {count + 1} train patients")
    selected = np.random.default_rng(seed).choice(ids, size=count + 1, replace=False)
    return [int(sid) for sid in selected]


@torch.no_grad()
def build_reference_bank(model, train_dataset, reference_ids, device):
    """Re-encode fixed train patients with current encoder; never use val/test rows."""
    lookup = {int(sid): i for i, sid in enumerate(train_dataset.sample_ids)}
    was_training = model.training
    model.eval()
    refs = []
    for sid in reference_ids:
        batch = collate_fn([train_dataset[lookup[int(sid)]]])
        state = model.encode_patient_state(
            sample_id=batch["sample_id"],
            tabular_features=batch["tabular_features"].to(device),
            tabular_observed_mask=batch["tabular_observed_mask"].to(device),
        )
        refs.append(state["f"].detach().squeeze(0))
    if was_training:
        model.train()
    return torch.stack(refs, dim=0)


def _training_references(sample_ids, reference_ids, reference_features, count):
    """Return J references for each target, excluding its own representation."""
    primary = list(range(count))
    rows = []
    for sid in sample_ids.tolist():
        used = [j for j in primary if reference_ids[j] != int(sid)]
        if len(used) < count:
            used.append(count)  # spare patient selected with the reference bank
        rows.append(reference_features[used])
    return torch.stack(rows, dim=0).detach()


def _augment_training_batch(model, batch, device):
    """Mask only observed training evidence; retain one real image view."""
    tab = batch["tabular_features"].to(device).clone()
    observed = batch["tabular_observed_mask"].to(device).clone()
    local, global_features, view_mask = model._get_image_features_by_id(
        batch["sample_id"])
    local, global_features, view_mask = (
        local.clone(), global_features.clone(), view_mask.clone())
    view_drop = float(getattr(model.train_config, "training_view_drop_prob", 0.0))
    feature_drop = float(getattr(model.train_config, "training_feature_drop_prob", 0.0))
    if not 0 <= view_drop < 1 or not 0 <= feature_drop < 1:
        raise ValueError("training drop probabilities must be in [0,1)")
    for row in range(view_mask.size(0)):
        present = torch.where(view_mask[row].bool())[0]
        if len(present) > 1 and view_drop > 0:
            drop = torch.rand(len(present), device=view_mask.device) < view_drop
            if drop.all():
                drop[torch.randint(len(present), (1,), device=view_mask.device)] = False
            view_mask[row, present[drop]] = 0
    missing_views = ~view_mask.bool()
    local[missing_views] = 0
    global_features[missing_views] = 0
    tab[missing_views] = 0
    observed[missing_views] = False
    if feature_drop > 0:
        hidden = ((torch.rand(tab.shape, device=device) < feature_drop)
                  & observed)
        tab[hidden] = 0
        observed[hidden] = False
    return tab, observed, (local, global_features, view_mask)


def train_one_epoch(model, loader, optimizer, device, epoch, reference_ids,
                    reference_features, reference_count, reference_lambda,
                    wandb_run=None, rank=0):
    model.train()
    total_loss, labels_all, preds_all = 0.0, [], []
    grad_steps = max(1, int(getattr(model.train_config, "gradient_accumulation_steps", 1)))
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        labels = batch["labels"].to(device)
        refs = _training_references(batch["sample_id"], reference_ids,
                                    reference_features, reference_count)
        tab, observed, image_override = _augment_training_batch(model, batch, device)
        out = model(
            sample_id=batch["sample_id"],
            tabular_features=tab,
            tabular_observed_mask=observed,
            labels=labels,
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            reference_queries=refs,
            reference_lambda=reference_lambda,
            image_features_override=image_override,
        )
        raw_loss = out["loss"]
        window_size = grad_steps
        if model.train_config.normalize_partial_window:
            window_start = (step // grad_steps) * grad_steps
            window_size = min(grad_steps, len(loader) - window_start)
        (raw_loss / window_size).backward()
        if (step + 1) % grad_steps == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach())
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(out["logits"].detach().argmax(-1).cpu().tolist())
        if rank == 0 and (step + 1) % 20 == 0:
            details = {k: round(float(v.detach()), 5) for k, v in out.get("loss_dict", {}).items()
                       if isinstance(v, torch.Tensor) and v.numel() == 1}
            logger.info("Epoch %s step %s/%s loss=%.5f losses=%s", epoch,
                        step + 1, len(loader), float(raw_loss.detach()), details)
            if wandb_run:
                wandb_run.log({"train/loss": float(raw_loss.detach()),
                               **{f"train/{k}": v for k, v in details.items()},
                               "epoch": epoch, "step": step})
    result = (total_loss / max(1, len(loader)),
              accuracy_score(labels_all, preds_all),
              f1_score(labels_all, preds_all, zero_division=0))
    if rank == 0:
        logger.info("[Train] epoch=%s loss=%.5f acc=%.5f f1=%.5f lambda=%.3f",
                    epoch, *result, float(model.effective_reference_lambda().detach())
                    if model.learn_reference_lambda else model.reference_lambda)
    return result
