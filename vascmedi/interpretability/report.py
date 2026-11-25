"""Run and plot model-path and module response analyses."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .._runtime import read_json
from ..models.vascmedi import FEATURE_MODULES
from ._plot import plot_panel
from .responses import build_reference_stability_banks, run_model_explanation


def explain(bundle, output, bootstrap):
    if bundle.cfg.comparison_mode != "full":
        raise ValueError("Module and pathway analyses require the mediator-based model")
    if bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous_features_dir = bundle.model.visual_features_dir
    try:
        bundle.model.visual_features_dir = str(bundle.cfg.visual_features_dir)
        banks = build_reference_stability_banks(
            bundle.model, bundle.training, bundle.reference_ids, bundle.references,
            bundle.cfg.reference_count, bundle.cfg.reference_seed, 3, bundle.device,
        )
        return run_model_explanation(
            bundle.model, bundle.testing, bundle.training, bundle.references, bundle.weight,
            str(output), bundle.device, n_bootstrap=bootstrap, include_features=False,
            include_hte=False, seed=bundle.cfg.seed, reference_stability_banks=banks,
            random_control=True, random_match_candidates=16, donor_sensitivity_shifts=3,
            test_features_dir=str(bundle.test_features),
        )
    finally:
        bundle.model.visual_features_dir = previous_features_dir


def plot_responses(source, output, endpoint, bootstrap=1000, seed=2026):
    import matplotlib.pyplot as plt

    source = Path(source)
    output = Path(output)
    patients = pd.read_csv(source / "model_explanation_patients.csv")
    modules = list(FEATURE_MODULES)
    values = patients[[f"IIE_{module}" for module in modules]].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values) or bootstrap < 1:
        raise ValueError("A positive bootstrap count and complete patient responses are required")
    samples = np.random.default_rng(seed).integers(0, len(values), size=(bootstrap, len(values)))
    low, high = np.quantile(values[samples].mean(axis=1), [0.025, 0.975], axis=0)
    mean = values.mean(axis=0)
    absolute = np.abs(values).mean(axis=0)
    shares = 100 * absolute / absolute.sum() if absolute.sum() else absolute
    data = pd.DataFrame({
        "module": [module.replace("_", " ").capitalize() for module in modules],
        "signed_iie": mean, "abs_share_pct": shares, "ci_low": low, "ci_high": high,
        "ci_crosses_zero": (low <= 0) & (high >= 0),
    }).sort_values("abs_share_pct", ascending=False)
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output / "module_responses.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.9, 3.2), layout="constrained")
    plot_panel(axis, data, endpoint.upper(), r"Mean signed module response ($\Delta P$)")
    figure.savefig(output / "module_responses.pdf", bbox_inches="tight")
    plt.close(figure)
    summary = read_json(source / "model_explanation_summary.json")["path"]
    with (output / "pathway_responses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "estimate", "ci_low", "ci_high"))
        for key in ("TE_model", "DE_model", "IE_model", "MR_ratio_of_means"):
            row = summary[key]
            writer.writerow((key, row.get("value", row.get("mean")), *row["ci95"]))
