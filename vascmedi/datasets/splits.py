"""Write patient partitions before fitting a model."""

import csv
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def _read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path} requires sample_id")
        return reader.fieldnames, list(reader)


def _stratified_split(label_map, seed):
    ids = sorted(label_map)
    labels = [label_map[pid] for pid in ids]
    validation_ratio, test_ratio = 0.1, 0.2
    held_ratio = validation_ratio + test_ratio
    # Match the original two-stage test_size arithmetic and rounding.
    train, held, _, held_labels = train_test_split(
        ids, labels, test_size=held_ratio, stratify=labels, random_state=seed,
    )
    validation, test = train_test_split(
        held, test_size=1.0 - validation_ratio / held_ratio,
        stratify=held_labels, random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def _load_split_manifest(path, cohort):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must contain train_ids, val_ids and test_ids")
    assigned = set()
    splits = {}
    for name, key in (("train", "train_ids"), ("validation", "val_ids"), ("test", "test_ids")):
        ids = manifest.get(key)
        if not isinstance(ids, list) or not ids or any(type(pid) is not int for pid in ids):
            raise ValueError(f"Split manifest requires a nonempty list of integer {key}")
        patients = set(ids)
        if len(patients) != len(ids) or assigned & patients:
            raise ValueError("Split manifest contains duplicate or overlapping patients")
        assigned.update(patients)
        splits[name] = ids
    if assigned != cohort:
        raise ValueError("Split manifest must exactly cover the labeled cohort")
    return splits


def prepare_splits(tabular, labels, output, seed, *, split_manifest=None):
    table_fields, rows = _read(tabular)
    label_fields, label_rows = _read(labels)
    label_key = "label"
    if label_key not in label_fields:
        label_columns = [key for key in label_fields if key != "sample_id"]
        if len(label_columns) != 1:
            raise ValueError("Label CSV requires a label column")
        label_key = label_columns[0]
    label_map = {int(row["sample_id"]): int(row[label_key]) for row in label_rows}
    if len(label_map) != len(label_rows) or set(label_map.values()) != {0, 1}:
        raise ValueError("Expected one binary label per patient and both classes")
    if not label_map.keys() <= {int(row["sample_id"]) for row in rows}:
        raise ValueError("Each labeled patient requires a feature row")
    if split_manifest is None:
        splits = _stratified_split(label_map, seed)
    else:
        splits = _load_split_manifest(split_manifest, label_map.keys())
    for name, patients in splits.items():
        if {label_map[i] for i in patients} != {0, 1}:
            raise ValueError(f"{name} requires both classes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, patients in splits.items():
        directory = output / name
        directory.mkdir()
        # Patient order determines seeded reference and minibatch sampling.
        (directory / "patient_order.json").write_text(
            json.dumps(patients) + "\n", encoding="utf-8",
        )
        patient_set = set(patients)
        for file, fields, contents in (
            ("tabular.csv", table_fields, rows),
            ("labels.csv", label_fields, label_rows),
        ):
            with (directory / file).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for row in contents if int(row["sample_id"]) in patient_set)
    return {name: len(patients) for name, patients in splits.items()}
