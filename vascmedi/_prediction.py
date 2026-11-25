"""Metrics and fixed-parameter inference shared by validation and test."""
import csv
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

logger = logging.getLogger(__name__)

def _classification_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    logits = torch.as_tensor(logits, dtype=torch.float32)
    prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = (prob >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, prob)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "auc": auc,
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "mcc": float(matthews_corrcoef(labels, preds)),
        "auprc": float(average_precision_score(labels, prob)) if len(np.unique(labels)) == 2 else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "loss": float(F.cross_entropy(logits, torch.as_tensor(labels)).item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, *, epoch, split, reference_queries,
             reference_lambda, output_path=None):
    """Score one fixed model and one fixed fusion coefficient."""
    model.eval()
    cur_all, ref_all, labels_all, patient_ids = [], [], [], []
    pure_logits_all = []
    query_all, reference_margin_std_all = [], []
    attention_all = {"layer1": [], "layer2": []}
    if reference_queries is None:
        raise ValueError("Validation/test requires a train-only reference bank")
    for batch in loader:
        out = model(
            sample_id=batch["sample_id"],
            tabular_features=batch["tabular_features"].to(device),
            tabular_observed_mask=batch["tabular_observed_mask"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            reference_queries=reference_queries,
            reference_lambda=1.0,
        )
        if out.get("pure_image_logits") is not None:
            pure_logits_all.append(out["pure_image_logits"].detach().float().cpu())
        cur_all.append(out["logits_cur"].detach().float().cpu())
        ref_all.append(out["logits_ref"].detach().float().cpu())
        query_all.append(out["F_query"].detach().float().cpu())
        if out.get("reference_margin_std") is not None:
            reference_margin_std_all.append(out["reference_margin_std"].detach().float().cpu())
        attention_pred = model.predict_from_states(
            out["F_query"], out["M_tokens"], reference_lambda=0.0,
            token_mask=out["M_token_mask"], return_attention=True,
        )
        if attention_pred["attention_cur"] is not None:
            for name, weights in attention_pred["attention_cur"].items():
                attention_all[name].append(weights.detach().float().mean(dim=1).squeeze(1).cpu())
        labels_all.extend(batch["labels"].tolist())
        patient_ids.extend(batch["sample_id"].tolist())
    cur, ref = torch.cat(cur_all), torch.cat(ref_all)
    chosen = float(reference_lambda)
    mixed = (1.0 - chosen) * cur + chosen * ref
    metrics = {**_classification_metrics(labels_all, mixed), "lambda": chosen}
    logger.info("[%s] AUC=%.5f ACC=%.5f F1=%.5f",
                split, metrics["auc"], metrics["acc"], metrics["f1"])
    if output_path:
        probs = torch.softmax((1.0 - chosen) * cur + chosen * ref, -1)[:, 1].numpy()
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "probability"])
            writer.writeheader()
            writer.writerows({"sample_id": int(sid), "label": int(y), "probability": float(p)}
                             for sid, y, p in zip(patient_ids, labels_all, probs))
    queries = torch.cat(query_all)
    references = reference_queries.detach().float().cpu()
    query_norm = F.normalize(queries, dim=-1)
    reference_norm = F.normalize(references, dim=-1)
    current_margin = cur[:, 1] - cur[:, 0]
    reference_margin = ref[:, 1] - ref[:, 0]
    diagnostics = {
        "epoch": epoch, "split": split, "n_patients": len(patient_ids),
        "reference_count": int(references.size(0)),
        "mean_query_to_reference_l2": float(torch.cdist(queries, references).mean()),
        "mean_query_to_reference_cosine_distance": float(
            (1.0 - query_norm @ reference_norm.T).mean()),
        "mean_query_feature_std_across_patients": float(
            queries.std(dim=0, unbiased=False).mean()),
        "mean_reference_feature_std": float(
            references.std(dim=0, unbiased=False).mean()),
        "mean_abs_current_reference_logit_margin_difference": float(
            (current_margin - reference_margin).abs().mean()),
        "mean_abs_current_reference_probability_difference": float(
            (torch.softmax(cur, dim=-1)[:, 1] - torch.softmax(ref, dim=-1)[:, 1]).abs().mean()),
        "individual_branch": _classification_metrics(labels_all, cur),
        "reference_branch": _classification_metrics(labels_all, ref),
        "mean_reference_logit_margin_std_across_J": (
            float(torch.cat(reference_margin_std_all).mean())
            if reference_margin_std_all else None),
    }
    if pure_logits_all:
        diagnostics["pure_image_auxiliary"] = _classification_metrics(labels_all, torch.cat(pure_logits_all))
        logger.info("[%s] pure image auxiliary metrics: %s", split, diagnostics["pure_image_auxiliary"])
    for name, chunks in attention_all.items():
        if chunks:
            weights = torch.cat(chunks)
            diagnostics[f"{name}_mean_token_weights"] = [float(v) for v in weights.mean(dim=0)]
            diagnostics[f"{name}_mean_attention_entropy"] = float(
                -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
            )
    logger.info("[%s] query diagnostics: %s", split, diagnostics)
    return metrics, diagnostics
