"""Separate patient files and training-only feature statistics."""
import csv
import hashlib
import json
import logging
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.vascmedi import FEATURE_MODULES

logger = logging.getLogger(__name__)

TABULAR_META_COLUMNS = {
    "sample_id",
    "eye",
    "part",
    "view_idx",
    "v_idx",
    "PatientID",
    "patient_id",
    "patientid",
}


def _validate_feature_schema(schema_path, feature_names):
    """Fail before model loading if CSV columns no longer match fixed module indices."""
    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    expected = schema.get("feature_names")
    if not isinstance(expected, list) or len(expected) != 58:
        raise ValueError("v5 feature schema must enumerate exactly 58 names")
    actual = list(feature_names)
    if actual != expected:
        first = next((j for j, (left, right) in enumerate(zip(actual, expected))
                      if left != right), min(len(actual), len(expected)))
        raise ValueError(
            f"Tabular feature order differs from v5 schema at index {first}: "
            f"CSV={actual[first:first + 1]!r}, schema={expected[first:first + 1]!r}. "
            "Supply an explicitly reviewed schema for a different 58-feature layout."
        )
    ranges = schema.get("module_ranges", {})
    for name, indices in FEATURE_MODULES.items():
        if ranges.get(name) != [indices[0], indices[-1] + 1]:
            raise ValueError(f"v5 feature schema does not match model module {name}")
    with open(schema_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"path": os.path.abspath(schema_path),
            "version": schema.get("schema_version"), "sha256": digest}


def _normalize_csv_key(key):
    if key is None:
        return None
    return key.lstrip("\ufeff").strip()


def _open_csv_reader(csv_path: str):
    csv_file = open(csv_path, "r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(csv_file)
    if not reader.fieldnames:
        csv_file.close()
        raise ValueError(f"CSV 文件为空或缺少表头: {csv_path}")
    reader.fieldnames = [_normalize_csv_key(k) for k in reader.fieldnames]
    return csv_file, reader


def _normalize_csv_row(row):
    normalized = {}
    for key, value in row.items():
        normalized[_normalize_csv_key(key)] = value.strip() if isinstance(value, str) else value
    return normalized


def _parse_float_or_nan(value):
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
    if value in ("", "nan", "NaN", "none", "None", "null", "NULL"):
        return np.nan
    return float(value)


def _format_label_distribution(label_counts):
    return ", ".join(f"{label}={label_counts.get(label, 0)}" for label in (0, 1))


def _compute_balanced_class_weights(labels, num_classes=2):
    label_array = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(label_array, minlength=num_classes).astype(np.float32)
    class_weights = np.ones(num_classes, dtype=np.float32)
    nonzero = class_counts > 0
    if np.any(nonzero):
        class_weights[nonzero] = class_counts.sum() / (nonzero.sum() * class_counts[nonzero])
    return torch.tensor(class_weights, dtype=torch.float32), class_counts.astype(int).tolist()


class StrokeDataset(Dataset):
    """
    脑梗预测多模态数据集
    每个样本包含：sample_id, 血管表格特征, 脑梗标签
    视觉特征由模型内部按 sample_id 从磁盘加载
    """

    def __init__(
        self,
        tabular_csv: str,       # 血管表格特征文件
        label_csv: str,         # 标签文件
        tokenizer,
        prompt_template: str = "Predict stroke risk based on fundus image: <|image_feature|>",
        context_length: int = 256,
        split: str = "train",
        training_dataset=None,
        *,
        patient_order=None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if (split == "train") != (training_dataset is None):
            raise ValueError("Validation and test require a training dataset for preprocessing")
        if training_dataset is not None and training_dataset.split != "train":
            raise ValueError("Preprocessing must come from the training split")
        self.split = split
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.context_length = context_length

        # 1. 加载表格特征 (支持一人多视图)
        # 按 v_idx 固定放入 4 个视图槽位，避免缺失视图时表格与图像视图错位。
        self.tabular_data = {}   # sample_id -> list of arrays (exactly 4)
        f, reader = _open_csv_reader(tabular_csv)
        try:
            if "sample_id" not in reader.fieldnames:
                raise ValueError(f"{tabular_csv} 缺少 sample_id 列。当前表头: {reader.fieldnames}")
            feat_keys = [k for k in reader.fieldnames if k not in TABULAR_META_COLUMNS]
            if not feat_keys:
                raise ValueError(f"{tabular_csv} 中未识别到任何表格特征列。当前表头: {reader.fieldnames}")
            for row in reader:
                row = _normalize_csv_row(row)
                sid = int(row["sample_id"])
                feat = np.array([_parse_float_or_nan(row[k]) for k in feat_keys], dtype=np.float32)
                if sid not in self.tabular_data:
                    self.tabular_data[sid] = [None] * 4
                try:
                    v_idx = int(row.get("v_idx", ""))
                except (TypeError, ValueError):
                    raise ValueError(f"Patient {sid} has a missing or invalid v_idx")
                if not 0 <= v_idx < 4:
                    raise ValueError(f"Patient {sid} has v_idx={v_idx}; expected 0..3")
                if self.tabular_data[sid][v_idx] is not None:
                    raise ValueError(f"Patient {sid} has duplicate v_idx={v_idx}")
                self.tabular_data[sid][v_idx] = feat
        finally:
            f.close()
        nan_feat = np.full(len(feat_keys), np.nan, dtype=np.float32)
        for sid, views in self.tabular_data.items():
            self.tabular_data[sid] = [feat if feat is not None else nan_feat.copy() for feat in views]
        self.tabular_dim = len(feat_keys)
        logger.info(
            f"Loaded tabular CSV: feature_dim={self.tabular_dim}, "
            f"features={feat_keys[:5]}{'...' if len(feat_keys) > 5 else ''}"
        )

        # 2. 加载标签
        self.labels = {}    # sample_id -> int
        f, reader = _open_csv_reader(label_csv)
        try:
            if "sample_id" not in reader.fieldnames:
                raise ValueError(f"{label_csv} 缺少 sample_id 列。当前表头: {reader.fieldnames}")
            label_key = "label"
            if label_key not in reader.fieldnames:
                non_id_fields = [k for k in reader.fieldnames if k != "sample_id"]
                if len(non_id_fields) == 1:
                    label_key = non_id_fields[0]
                else:
                    raise ValueError(f"{label_csv} 缺少标签列。当前表头: {reader.fieldnames}")
            for row in reader:
                row = _normalize_csv_row(row)
                sid = int(row["sample_id"])
                label = int(row[label_key])
                if sid in self.labels or label not in (0, 1):
                    raise ValueError("Labels must contain one binary value per patient")
                self.labels[sid] = label
        finally:
            f.close()
        logger.info(f"Loaded label CSV: label_column={label_key}")

        if not self.tabular_data or self.labels.keys() != self.tabular_data.keys():
            raise ValueError("Feature and label CSVs must contain the same nonempty patient set")
        if patient_order is None:
            self.sample_ids = sorted(self.labels)
        else:
            if (not isinstance(patient_order, list)
                    or any(type(pid) is not int for pid in patient_order)
                    or len(patient_order) != len(self.labels)
                    or set(patient_order) != self.labels.keys()):
                raise ValueError("Patient order must list each patient in the split exactly once")
            self.sample_ids = list(patient_order)
        self.sample_labels = [self.labels[sid] for sid in self.sample_ids]
        self.split_label_counts = Counter(self.sample_labels)
        if len(self.split_label_counts) != 2:
            raise ValueError(f"{split} requires both classes to report AUC")
        self.feature_names = feat_keys
        if training_dataset is None:
            raw = np.stack([view for sid in self.sample_ids for view in self.tabular_data[sid]])
            observed = ~np.all(np.isnan(raw), axis=0)
            if not np.any(observed):
                raise ValueError("All training features are missing")
            self.mean = np.zeros(len(feat_keys), dtype=np.float32)
            self.std = np.ones(len(feat_keys), dtype=np.float32)
            self.mean[observed] = np.nanmean(raw[:, observed], axis=0)
            std = (np.nanstd(raw[:, observed], axis=0) + 1e-6).astype(np.float32)
            self.std[observed] = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
            self.train_label_counts = self.split_label_counts
            self.class_weights, self.train_class_count_list = _compute_balanced_class_weights(self.sample_labels)
        else:
            if feat_keys != training_dataset.feature_names:
                raise ValueError(f"{split} feature columns differ from training columns")
            if set(self.sample_ids) & set(training_dataset.sample_ids):
                raise ValueError(f"Patient overlap between train and {split}")
            self.mean = training_dataset.mean.copy()
            self.std = training_dataset.std.copy()

        logger.info(
            f"[{split}] {len(self.sample_ids)} patients, "
            f"tabular_dim={len(self.mean)}, "
            f"labels: {_format_label_distribution(self.split_label_counts)}"
        )
        if split == "train":
            logger.info(
                "Training split class weights: [%s]",
                ", ".join(f"{w:.4f}" for w in self.class_weights.tolist()),
            )
        elif self.split_label_counts.get(1, 0) < 3:
            logger.warning(
                "[%s] positive samples are very few (%d). Metrics such as AUC/F1 may remain unstable.",
                split,
                self.split_label_counts.get(1, 0),
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        # 5. 准备 4 个视图的表格特征 (缺失补零)
        # [4, D]
        tab_list = self.tabular_data[sid]
        final_tab = np.zeros((4, len(self.mean)), dtype=np.float32)
        observed = np.stack([np.isfinite(feat) for feat in tab_list])

        for i, feat in enumerate(tab_list):
            feat = np.where(np.isnan(feat), self.mean, feat)
            final_tab[i] = (feat - self.mean) / self.std

        # Tokenize prompt
        encoded = self.tokenizer(
            self.prompt_template,
            truncation=True,
            max_length=self.context_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "sample_id":       torch.tensor(sid, dtype=torch.int),
            "tabular_features": torch.tensor(final_tab, dtype=torch.float32),
            "tabular_observed_mask": torch.tensor(observed, dtype=torch.bool),
            "labels":          torch.tensor(self.labels[sid], dtype=torch.long),
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
        }


def collate_fn(batch):
    return {
        "sample_id":       torch.stack([b["sample_id"]       for b in batch]),
        "tabular_features": torch.stack([b["tabular_features"] for b in batch]),
        "tabular_observed_mask": torch.stack([b["tabular_observed_mask"] for b in batch]),
        "labels":          torch.stack([b["labels"]          for b in batch]),
        "input_ids":       torch.stack([b["input_ids"]       for b in batch]),
        "attention_mask":  torch.stack([b["attention_mask"]  for b in batch]),
    }
