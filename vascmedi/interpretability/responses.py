"""Pathway decomposition and phenotype module interventions."""
import csv
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..datasets.retinal import collate_fn
from .._learning import build_reference_bank

logger = logging.getLogger(__name__)

def _mean_ci(values, seed, n_bootstrap):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    boot = x[picks].mean(axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "sd_across_patients": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "q10": float(np.quantile(x, 0.10)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)), "q90": float(np.quantile(x, 0.90)),
            "positive_fraction": float(np.mean(x > 0))}


def _sign_flip_mean_p(values, seed, n_permutations=1000):
    """Exploratory paired sign-flip test under a symmetric zero-response null."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (n_permutations + 1))


def _benjamini_hochberg(p_values):
    """Return FDR adjusted q values in the original hypothesis order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _matched_random_columns(tab, donor_tab, module_columns, valid_cell_mask,
                            rng, n_candidates=32):
    """Choose a label-blind random subset with the same size and similar input L2.

    Both target and control replace values with the same real training donor.
    The matched feature subset is selected only from other named table columns.
    """
    n_features = int(tab.shape[-1])
    module_columns = [int(j) for j in module_columns]
    other_columns = np.asarray(
        [j for j in range(n_features) if j not in set(module_columns)], dtype=np.int64)
    if len(other_columns) < len(module_columns):
        raise ValueError("Not enough non-module features for a same-size random control")
    valid = torch.as_tensor(valid_cell_mask, dtype=torch.bool, device=tab.device)
    if valid.ndim == 3 and valid.size(0) == 1:
        valid = valid[0]
    if valid.shape != tab.shape[-2:]:
        raise ValueError("valid cell mask must match the tabular [view,feature] shape")
    delta = (tab - donor_tab).detach().float().reshape(-1, n_features)
    delta = delta.masked_fill(~valid.reshape(-1, n_features), 0.0)
    feature_energy = delta.square().sum(dim=0).cpu().numpy()
    target_l2 = float(np.sqrt(feature_energy[module_columns].sum()))
    best_columns, best_l2, best_error = None, None, float("inf")
    for _ in range(max(1, int(n_candidates))):
        candidate = rng.choice(other_columns, size=len(module_columns), replace=False)
        candidate_l2 = float(np.sqrt(feature_energy[candidate].sum()))
        error = abs(candidate_l2 - target_l2)
        if error < best_error:
            best_columns = sorted(int(j) for j in candidate)
            best_l2, best_error = candidate_l2, error
    tolerance = max(0.25 * target_l2, 0.1)
    return {
        "columns": best_columns,
        "target_input_l2": target_l2,
        "random_input_l2": best_l2,
        "absolute_match_error": best_error,
        "relative_match_error": best_error / target_l2 if target_l2 > 1e-8 else None,
        "eligible_match": (bool(valid[:, module_columns].any().item())
                           and bool(valid[:, best_columns].any().item())
                           and best_error <= tolerance),
    }


def _rank_correlation(a, b):
    """Spearman correlation of eight module magnitudes; undefined for all ties."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        out = np.empty(len(x), dtype=float)
        sorted_x = x[order]
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and np.isclose(sorted_x[end], sorted_x[start], atol=1e-12, rtol=0):
                end += 1
            out[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return out
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _reference_stability_summary(rows, module_names, banks, reference_lambda,
                                 n_bootstrap, seed):
    """Test prediction and eight-module ranking stability across train-only banks."""
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    n = len(rows)
    base_prob = np.asarray([r["refset_0_prob"] for r in rows], dtype=float)
    base_effect = np.asarray([[r[f"IIE_{name}"] for name in module_names]
                              for r in rows], dtype=float)
    def magnitude_by_module(effect):
        observed = np.isfinite(effect)
        counts = observed.sum(axis=0)
        values = np.where(observed, np.abs(effect), 0.0).sum(axis=0) / np.maximum(counts, 1)
        values[counts == 0] = np.nan
        return values

    base_magnitude = magnitude_by_module(base_effect)
    base_order = sorted((j for j in range(len(module_names)) if np.isfinite(base_magnitude[j])),
                        key=lambda j: (-base_magnitude[j], module_names[j]))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n, size=(n_bootstrap, n))
    result = {
        "scope": "final_checkpoint_train_reference_resampling_not_causal_effect",
        "fixed_lambda": float(reference_lambda), "reference_count": len(banks[0]["ids"]),
        "reference_branch_active": bool(reference_lambda > 0),
        "lambda_zero_note": (
            "At lambda=0 the final predictor ignores reference queries, so identical "
            "predictions and rankings across banks are structural, not evidence of stability."
            if reference_lambda == 0 else None),
        "bootstrap_unit": "test_patient", "bootstrap_resamples": int(n_bootstrap),
        "module_score": "mean_absolute_input_module_intervention_response",
        "sets": [],
    }
    for bank in banks:
        idx = int(bank["set_index"])
        prob = np.asarray([r[f"refset_{idx}_prob"] for r in rows], dtype=float)
        effect = base_effect if idx == 0 else np.asarray(
            [[r[f"refset_{idx}_IIE_{name}"] for name in module_names]
             for r in rows], dtype=float)
        magnitude = magnitude_by_module(effect)
        order = sorted((j for j in range(len(module_names)) if np.isfinite(magnitude[j])),
                       key=lambda j: (-magnitude[j], module_names[j]))
        rho = _rank_correlation(base_magnitude, magnitude)
        rho_boot = []
        for patient_indices in picks:
            base_boot = magnitude_by_module(base_effect[patient_indices])
            bank_boot = magnitude_by_module(effect[patient_indices])
            boot_rho = _rank_correlation(base_boot, bank_boot)
            if boot_rho is not None:
                rho_boot.append(boot_rho)
        top_base = set(base_order[:3])
        pred = (prob >= 0.5).astype(np.int64)
        auc = float(roc_auc_score(labels, prob)) if len(np.unique(labels)) == 2 else None
        record = {
            "set_index": idx, "seed": int(bank["seed"]),
            "train_patient_ids": [int(x) for x in bank["ids"]],
            "n_test_patients": n,
            "auc": auc, "acc": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "prediction_absolute_difference_from_primary": _mean_ci(
                np.abs(prob - base_prob), seed, n_bootstrap),
            "module_mean_absolute_responses": {
                name: float(magnitude[j]) if np.isfinite(magnitude[j]) else None
                for j, name in enumerate(module_names)},
            "module_ranking": [module_names[j] for j in order],
            "module_rank_spearman_vs_primary": rho,
            "module_rank_spearman_bootstrap_ci95": (
                [float(v) for v in np.quantile(rho_boot, [0.025, 0.975])]
                if rho_boot else [None, None]),
            "module_rank_valid_bootstrap_fraction": len(rho_boot) / n_bootstrap,
            "top_module_matches_primary": (order[0] == base_order[0]
                                            if order and base_order else None),
            "top3_overlap_with_primary": (len(top_base & set(order[:3])) / min(3, len(base_order))
                                           if base_order else None),
        }
        result["sets"].append(record)
    return result


def _random_control_summary(rows, module_names, n_bootstrap, seed,
                            n_candidates):
    """Paired comparison of module response against same-size L2-matched random inputs."""
    result = {
        "scope": "label_blind_model_intervention_faithfulness_control",
        "selection": "same donor, same feature count, closest input L2 among label-blind random subsets",
        "candidate_subsets_per_patient_module": int(n_candidates),
        "eligible_match_rule": "absolute input L2 difference <= max(0.25 * target L2, 0.1)",
        "bootstrap_unit": "test_patient", "bootstrap_resamples": int(n_bootstrap),
        "modules": {},
    }
    per_patient_differences = [[] for _ in rows]
    for name in module_names:
        eligible = [r for r in rows if r[f"random_{name}_eligible_match"]]
        actual_all = [abs(r[f"IIE_{name}"]) for r in rows
                      if np.isfinite(r[f"IIE_{name}"])]
        random_all = [abs(r[f"random_{name}_IIE"]) for r in rows
                      if np.isfinite(r[f"random_{name}_IIE"])]
        paired = [abs(r[f"IIE_{name}"]) - abs(r[f"random_{name}_IIE"])
                  for r in eligible]
        for i, row in enumerate(rows):
            if row[f"random_{name}_eligible_match"]:
                per_patient_differences[i].append(
                    abs(row[f"IIE_{name}"]) - abs(row[f"random_{name}_IIE"]))
        paired_stats = _mean_ci(paired, seed, n_bootstrap)
        lo = paired_stats["ci95"][0]
        result["modules"][name] = {
            "n_patients": len(rows), "n_input_strength_matched": len(eligible),
            "match_fraction": len(eligible) / len(rows),
            "actual_mean_absolute_response_all": (
                float(np.mean(actual_all)) if actual_all else None),
            "random_mean_absolute_response_all": (
                float(np.mean(random_all)) if random_all else None),
            "paired_absolute_response_difference_matched": paired_stats,
            "actual_exceeds_random_fraction_matched": (
                float(np.mean(np.asarray(paired) > 0)) if paired else None),
            "input_l2_match_absolute_error": _mean_ci(
                [r[f"random_{name}_match_error"] for r in rows], seed, n_bootstrap),
            "exploratory_faithfulness_flag": bool(lo is not None and lo > 0),
        }
    patient_means = [float(np.mean(values)) for values in per_patient_differences if values]
    result["across_modules_patient_mean_paired_difference"] = _mean_ci(
        patient_means, seed, n_bootstrap)
    result["n_patients_with_at_least_one_matched_module"] = len(patient_means)
    return result


@torch.no_grad()
def build_reference_stability_banks(model, train_dataset, primary_ids,
                                    primary_queries, count, seed, n_sets, device):
    """Independent uniform train-only reference samples under the final encoder."""
    if n_sets < 3:
        raise ValueError("Reference stability requires at least three reference sets")
    all_train_ids = np.asarray(train_dataset.sample_ids, dtype=np.int64)
    if len(all_train_ids) < count:
        raise ValueError("Reference count exceeds training-patient count")
    banks = [{"set_index": 0, "seed": int(seed),
              "ids": [int(x) for x in primary_ids[:count]],
              "queries": primary_queries.detach()}]
    primary_set = set(banks[0]["ids"])
    for set_index in range(1, n_sets):
        draw_seed = int(seed + 1009 * set_index)
        draw = np.random.default_rng(draw_seed).choice(
            all_train_ids, size=count, replace=False)
        ids = [int(x) for x in draw]
        if set(ids) == primary_set:
            draw_seed += 1
            ids = [int(x) for x in np.random.default_rng(draw_seed).choice(
                all_train_ids, size=count, replace=False)]
        queries = build_reference_bank(model, train_dataset, ids, device)
        banks.append({"set_index": set_index, "seed": draw_seed,
                      "ids": ids, "queries": queries.detach()})
    return banks


@torch.no_grad()
def run_model_explanation(model, test_dataset, train_dataset, reference_queries,
                          reference_lambda, output_dir, device, n_bootstrap=500,
                          include_features=True, include_hte=True, seed=2026,
                          reference_stability_banks=None, random_control=False,
                          random_match_candidates=32, donor_sensitivity_shifts=3,
                          *, test_features_dir=None):
    """Model response analysis on the exact fused predictor used for testing.

    Baselines are real negative training patients. These are model interventions,
    not identified clinical causal effects or treatment HTE.
    With separate caches, the caller sets the training cache before entry and
    restores the original cache after the analysis.
    """
    model.eval()
    controls = [int(sid) for sid in train_dataset.sample_ids
                if train_dataset.labels[int(sid)] == 0]
    if not controls:
        raise ValueError("Model-path contrasts require negative training baselines")
    rng = np.random.default_rng(seed)
    baseline_ids = [int(x) for x in rng.choice(controls, size=min(16, len(controls)),
                                                replace=False)]
    donor_sensitivity_shifts = min(int(donor_sensitivity_shifts), len(baseline_ids))
    if donor_sensitivity_shifts < 1:
        raise ValueError("donor_sensitivity_shifts must be positive")
    train_lookup = {int(sid): i for i, sid in enumerate(train_dataset.sample_ids)}
    baseline_states = {}
    baseline_tabs = {}
    baseline_observed_cells = {}
    for sid in baseline_ids:
        batch = collate_fn([train_dataset[train_lookup[sid]]])
        baseline_tabs[sid] = batch["tabular_features"].to(device)
        baseline_observed_cells[sid] = batch["tabular_observed_mask"][0].to(device)
        baseline_states[sid] = model.encode_patient_state(
            sample_id=batch["sample_id"], tabular_features=baseline_tabs[sid],
            tabular_observed_mask=batch["tabular_observed_mask"].to(device))

    # A second, local baseline checks whether cross-label donor replacement
    # exaggerates a module response. Labels are used only for this post hoc
    # explanation analysis, never for prediction or reference selection.
    same_label_pool = {0: [], 1: []}
    for sid in train_dataset.sample_ids:
        raw_views = np.stack(train_dataset.tabular_data[int(sid)])
        observed_cells = np.isfinite(raw_views)
        standardized = (
            np.where(np.isnan(raw_views), train_dataset.mean, raw_views)
            - train_dataset.mean
        ) / train_dataset.std
        same_label_pool[int(train_dataset.labels[int(sid)])].append(
            (int(sid), standardized.astype(np.float32), observed_cells)
        )

    stability_banks = list(reference_stability_banks or [])
    if stability_banks and (len(stability_banks) < 3 or
                            int(stability_banks[0]["queries"].shape[0]) != int(reference_queries.shape[0])):
        raise ValueError("Reference stability needs primary plus at least two equal-J banks")
    if random_match_candidates < 1:
        raise ValueError("random_match_candidates must be positive")

    def score(f, mediator, token_mask, refs=None, return_margin=False):
        out = model.predict_from_states(
            f, mediator, reference_queries=reference_queries if refs is None else refs,
            reference_lambda=reference_lambda, token_mask=token_mask)
        probability = float(out["probs"][0, 1].item())
        if return_margin:
            margin = float((out["logits"][0, 1] - out["logits"][0, 0]).item())
            return probability, margin
        return probability

    rows = []
    module_names = list(model.feature_modules)
    feature_names = list(test_dataset.feature_names)
    if test_features_dir is not None:
        # Training donor states have been encoded. All remaining image reads
        # are for test patients, including their vascular interventions.
        model.visual_features_dir = test_features_dir
    for index in range(len(test_dataset)):
        batch = collate_fn([test_dataset[index]])
        sid = int(batch["sample_id"][0])
        tab = batch["tabular_features"].to(device)
        state = model.encode_patient_state(
            sample_id=batch["sample_id"], tabular_features=tab,
            tabular_observed_mask=batch["tabular_observed_mask"].to(device))
        f1, m1, mask1 = state["f"], state["M"], state["token_mask"]
        donor_id = baseline_ids[index % len(baseline_ids)]
        donor = baseline_states[donor_id]
        donor_tab = baseline_tabs[donor_id]
        donor_observed_cells = baseline_observed_cells[donor_id]
        f0, m0, mask0 = donor["f"], donor["M"], donor["token_mask"]
        p11, a11 = score(f1, m1, mask1, return_margin=True)
        p00, a00 = score(f0, m0, mask0, return_margin=True)
        p10, a10 = score(f1, m0, mask0, return_margin=True)
        p01, a01 = score(f0, m1, mask1, return_margin=True)
        row = {
            "sample_id": sid, "label": int(batch["labels"][0]),
            "baseline_id": donor_id, "p11": p11, "p00": p00, "p10": p10,
            "TE_model": p11 - p00, "DE_model": p10 - p00,
            "IE_model": p11 - p10,
            "logit_margin_11": a11, "logit_margin_00": a00,
            "logit_margin_10": a10,
            "TE_logit": a11 - a00, "DE_logit": a10 - a00,
            "IE_logit": a11 - a10,
            "p01": p01, "logit_margin_01": a01,
            "DE_reverse_model": p11 - p01,
            "IE_reverse_model": p01 - p00,
            "IE_order_difference": (p11 - p10) - (p01 - p00),
        }
        row["baseline_shift_0_id"] = donor_id
        row["baseline_shift_0_TE"] = row["TE_model"]
        row["baseline_shift_0_DE"] = row["DE_model"]
        row["baseline_shift_0_IE"] = row["IE_model"]
        for shift in range(1, donor_sensitivity_shifts):
            alternate_id = baseline_ids[(index + shift) % len(baseline_ids)]
            alternate = baseline_states[alternate_id]
            alternate_p00 = score(
                alternate["f"], alternate["M"], alternate["token_mask"])
            alternate_p10 = score(
                f1, alternate["M"], alternate["token_mask"])
            row[f"baseline_shift_{shift}_id"] = alternate_id
            row[f"baseline_shift_{shift}_TE"] = p11 - alternate_p00
            row[f"baseline_shift_{shift}_DE"] = alternate_p10 - alternate_p00
            row[f"baseline_shift_{shift}_IE"] = p11 - alternate_p10
        same_candidates = same_label_pool[row["label"]]
        target_tab = tab[0].detach().float().cpu().numpy()
        target_view_mask = state["v_mask"].bool()
        donor_view_mask = donor["v_mask"].bool()
        target_observed_cells = (
            batch["tabular_observed_mask"][0].to(device)
            & target_view_mask[0].unsqueeze(-1)
        )
        target_cells_numpy = target_observed_cells.cpu().numpy()
        nearest_id, nearest_values, nearest_cells = min(
            same_candidates,
            key=lambda item: (
                not bool(np.any(target_cells_numpy & item[2])),
                float(np.mean((item[1][target_cells_numpy & item[2]]
                               - target_tab[target_cells_numpy & item[2]]) ** 2))
                if np.any(target_cells_numpy & item[2]) else float("inf"),
            ),
        )
        same_tab = torch.as_tensor(nearest_values, device=tab.device,
                                   dtype=tab.dtype).unsqueeze(0)
        row["same_label_nearest_baseline_id"] = nearest_id
        local_shared_cells = target_observed_cells & torch.as_tensor(
            nearest_cells, device=device, dtype=torch.bool)
        row["same_label_nearest_input_l2"] = (
            float(np.linalg.norm((nearest_values - target_tab)[local_shared_cells.cpu().numpy()]))
            if local_shared_cells.any() else float("nan")
        )
        if stability_banks:
            row["refset_0_prob"] = p11
            for bank in stability_banks[1:]:
                row[f"refset_{bank['set_index']}_prob"] = score(
                    f1, m1, mask1, refs=bank["queries"])
        donor_shared_cells = (
            target_observed_cells & donor_observed_cells
            & donor_view_mask[0].unsqueeze(-1)
        )
        row["target_valid_views"] = int(target_view_mask.sum().item())
        row["donor_valid_views"] = int(donor_view_mask.sum().item())
        row["target_observed_cells"] = int(target_observed_cells.sum().item())
        row["shared_observed_cells"] = int(donor_shared_cells.sum().item())
        row["same_label_nearest_shared_observed_cells"] = int(local_shared_cells.sum().item())
        row["shared_valid_view_slots"] = int(donor_shared_cells.any(dim=1).sum().item())
        row["same_label_nearest_shared_view_slots"] = int(
            local_shared_cells.any(dim=1).sum().item())
        denom = abs(row["IE_model"]) + abs(row["DE_model"])
        row["MR_abs_individual"] = abs(row["IE_model"]) / denom if denom > 1e-8 else float("nan")

        # Every table intervention starts at named input columns. LLaMA is reused
        # exactly because the current query depends on image/prompt, not the table.
        language_cache = state.get("language_cache")
        if language_cache is None:
            raise RuntimeError("v19 model must return language_cache for exact table interventions")

        def changed_state(changed_tab):
            return model.encode_patient_state(
                sample_id=batch["sample_id"], tabular_features=changed_tab,
                tabular_observed_mask=batch["tabular_observed_mask"].to(device),
                fixed_language_state=language_cache)

        def changed_score(changed_tab):
            changed = changed_state(changed_tab)
            return score(f1, changed["M"], changed["token_mask"])

        def replace_observed_cells(source_tab, columns, valid_cells):
            changed_tab = tab.clone()
            for column in ([columns] if isinstance(columns, int) else columns):
                valid_views = valid_cells[:, column]
                changed_tab[0, valid_views, column] = source_tab[0, valid_views, column]
            return changed_tab

        joint_cols = list(range(tab.size(-1)))
        joint_cells = int(donor_shared_cells.sum().item())
        row["vascular_joint_eligible_cells"] = joint_cells
        row["vascular_joint_response"] = (p11 - changed_score(
            replace_observed_cells(donor_tab, joint_cols, donor_shared_cells))
            if joint_cells else float("nan"))
        for module_index, name in enumerate(module_names):
            cols = model.feature_modules[name]
            n_donor = int(donor_shared_cells[:, cols].sum().item())
            n_local = int(local_shared_cells[:, cols].sum().item())
            row[f"IIE_{name}_eligible_cells"] = n_donor
            row[f"IIE_same_label_nearest_{name}_eligible_cells"] = n_local
            changed = None
            if n_donor:
                changed_tab = replace_observed_cells(donor_tab, cols, donor_shared_cells)
                changed = changed_state(changed_tab)
                row[f"IIE_{name}"] = p11 - score(f1, changed["M"], changed["token_mask"])
            else:
                row[f"IIE_{name}"] = float("nan")
            if n_local:
                same_changed_tab = replace_observed_cells(same_tab, cols, local_shared_cells)
                row[f"IIE_same_label_nearest_{name}"] = p11 - changed_score(same_changed_tab)
            else:
                row[f"IIE_same_label_nearest_{name}"] = float("nan")
            for bank in stability_banks[1:]:
                set_index = bank["set_index"]
                row[f"refset_{set_index}_IIE_{name}"] = (
                    row[f"refset_{set_index}_prob"] - score(
                        f1, changed["M"], changed["token_mask"], refs=bank["queries"])
                    if changed is not None else float("nan"))
            if random_control:
                blind_rng = np.random.default_rng(
                    int(seed + sid * 9973 + module_index * 101))
                match = _matched_random_columns(
                    tab, donor_tab, cols, donor_shared_cells, blind_rng,
                    n_candidates=random_match_candidates)
                if match["eligible_match"]:
                    random_tab = replace_observed_cells(
                        donor_tab, match["columns"], donor_shared_cells)
                    row[f"random_{name}_IIE"] = p11 - changed_score(random_tab)
                else:
                    row[f"random_{name}_IIE"] = float("nan")
                row[f"random_{name}_columns"] = json.dumps(match["columns"])
                row[f"random_{name}_target_input_l2"] = match["target_input_l2"]
                row[f"random_{name}_input_l2"] = match["random_input_l2"]
                row[f"random_{name}_match_error"] = match["absolute_match_error"]
                row[f"random_{name}_eligible_match"] = int(match["eligible_match"])

        if include_features:
            for feature_index, name in enumerate(feature_names):
                valid = donor_shared_cells[:, feature_index]
                row[f"feature_{feature_index}_IIE_eligible_cells"] = int(valid.sum().item())
                if valid.any():
                    changed_tab = replace_observed_cells(
                        donor_tab, feature_index, donor_shared_cells)
                    row[f"feature_{feature_index}_IIE"] = p11 - changed_score(changed_tab)
                else:
                    row[f"feature_{feature_index}_IIE"] = float("nan")

        if include_hte:
            for feature_index, name in enumerate(feature_names):
                valid = target_observed_cells[:, feature_index]
                count = int(valid.sum().item())
                row[f"feature_{feature_index}_perturb_eligible_cells"] = count
                if not count:
                    for suffix in ("plus_response", "minus_response",
                                   "plus_clip_fraction", "minus_clip_fraction"):
                        row[f"feature_{feature_index}_{suffix}"] = float("nan")
                    continue
                plus = tab.clone()
                minus = tab.clone()
                plus[0, valid, feature_index] = tab[0, valid, feature_index] + (3 - tab[0, valid, feature_index]).clamp(0, 0.5)
                minus[0, valid, feature_index] = tab[0, valid, feature_index] - (tab[0, valid, feature_index] + 3).clamp(0, 0.5)
                row[f"feature_{feature_index}_plus_response"] = changed_score(plus) - p11
                row[f"feature_{feature_index}_minus_response"] = p11 - changed_score(minus)
                row[f"feature_{feature_index}_plus_clip_fraction"] = float(
                    ((plus[0, valid, feature_index] - tab[0, valid, feature_index]) < 0.5-1e-6).float().mean().item())
                row[f"feature_{feature_index}_minus_clip_fraction"] = float(
                    ((tab[0, valid, feature_index] - minus[0, valid, feature_index]) < 0.5-1e-6).float().mean().item())
        rows.append(row)
        if (index + 1) % 25 == 0:
            logger.info("[model explanation] %s/%s test patients", index + 1, len(test_dataset))

    path = os.path.join(output_dir, "model_explanation_patients.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scope": "model_prediction_response_not_clinical_causal_effect",
        "multiple_testing_note": "The 58 feature HTE responses have exploratory paired sign-flip p values and Benjamini-Hochberg q values across both directions (116 tests); sign-flip symmetry is an assumption, not clinical causal identification.",
        "path_population": "positive_test_patients_vs_real_negative_training_patients",
        "cross_patient_path_note": "The p10 contrast pairs a test patient's query with a real training donor's mediator; this synthetic pairing may be outside observed support.",
        "tabular_donor_policy": "replace only target/donor jointly observed cells in valid visual view slots; missing-value imputations are never intervened on",
        "local_donor_policy": "nearest standardized tabular patient with the same label in the training split; post hoc sensitivity only",
        "reference_lambda": float(reference_lambda),
        "reference_count": int(reference_queries.shape[0]),
        "baseline_ids": baseline_ids,
        "donor_sensitivity_shifts": donor_sensitivity_shifts,
        "bootstrap_scope_note": "patient bootstrap intervals condition on the selected baseline assignment; donor assignment sensitivity is reported separately",
        "view_coverage": {
            "mean_target_valid_views": float(np.mean([r["target_valid_views"] for r in rows])),
            "mean_donor_valid_views": float(np.mean([r["donor_valid_views"] for r in rows])),
            "mean_shared_valid_view_slots": float(np.mean([r["shared_valid_view_slots"] for r in rows])),
            "mean_target_observed_cells": float(np.mean([r["target_observed_cells"] for r in rows])),
            "mean_shared_observed_cells": float(np.mean([r["shared_observed_cells"] for r in rows])),
            "fraction_with_donor_shared_slot": float(np.mean(
                [r["shared_valid_view_slots"] > 0 for r in rows])),
            "fraction_with_same_label_shared_slot": float(np.mean(
                [r["same_label_nearest_shared_view_slots"] > 0 for r in rows])),
        },
        "bootstrap_resamples": int(n_bootstrap),
        "vascular_joint_intervention": _mean_ci([r["vascular_joint_response"] for r in rows], seed, n_bootstrap),
        "vascular_joint_mean_absolute_response": float(np.nanmean([abs(r["vascular_joint_response"]) for r in rows])),
        "path": {}, "path_all_patients": {}, "modules": {}, "features": {}, "hte": {},
    }
    positive_rows = [r for r in rows if r["label"] == 1]
    if not positive_rows:
        raise ValueError("MR requires at least one positive test patient")
    for key in ("TE_model", "DE_model", "IE_model", "DE_reverse_model",
                "IE_reverse_model", "IE_order_difference", "TE_logit",
                "DE_logit", "IE_logit", "MR_abs_individual"):
        summary["path"][key] = _mean_ci([r[key] for r in positive_rows], seed, n_bootstrap)
        summary["path_all_patients"][key] = _mean_ci([r[key] for r in rows], seed, n_bootstrap)
    te = np.asarray([r["TE_model"] for r in positive_rows], dtype=np.float64)
    de = np.asarray([r["DE_model"] for r in positive_rows], dtype=np.float64)
    ie = np.asarray([r["IE_model"] for r in positive_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(positive_rows), size=(n_bootstrap, len(positive_rows)))
    boot_te, boot_ie = te[picks].mean(1), ie[picks].mean(1)
    valid_boot = np.abs(boot_te) >= 0.01
    boot_mr = boot_ie[valid_boot] / boot_te[valid_boot]
    mean_te, mean_de, mean_ie = float(te.mean()), float(de.mean()), float(ie.mean())
    te_ci = np.quantile(boot_te, [0.025, 0.975])
    te_crosses_zero = bool(te_ci[0] <= 0 <= te_ci[1])
    de_ci = summary["path"]["DE_model"]["ci95"]
    de_crosses_zero = bool(de_ci[0] <= 0 <= de_ci[1])
    structural = float(reference_lambda) >= 1.0 - 1e-8
    mr_valid = (abs(mean_te) >= 0.01 and not te_crosses_zero and not de_crosses_zero
                and not structural and
                mean_ie * mean_te >= -1e-8 and mean_de * mean_te >= -1e-8)
    if structural:
        mr_status = "structural_direct_path_removed_at_lambda_one"
    elif abs(mean_te) < 0.01:
        mr_status = "near_zero_total_response"
    elif te_crosses_zero:
        mr_status = "unstable_total_response_ci_crosses_zero"
    elif de_crosses_zero:
        mr_status = "query_path_indistinguishable_from_zero"
    elif mean_ie * mean_te < -1e-8 or mean_de * mean_te < -1e-8:
        mr_status = "suppressor_opposite_path_signs"
    else:
        mr_status = "valid_model_path_ratio"
    summary["path"]["MR_ratio_of_means"] = {
        "value": float(mean_ie / mean_te) if mr_valid else None,
        "raw_value": float(mean_ie / mean_te) if abs(mean_te) >= 0.01 else None,
        "valid": bool(mr_valid),
        "status": mr_status,
        "total_response_ci95": [float(v) for v in te_ci],
        "direct_response_ci95": [float(v) for v in de_ci],
        "valid_bootstrap_fraction": float(np.mean(valid_boot)),
        "ci95": [float(v) for v in np.quantile(boot_mr, [0.025, 0.975])]
        if mr_valid and len(boot_mr) else [None, None],
        "raw_bootstrap_ci95": [float(v) for v in np.quantile(boot_mr, [0.025, 0.975])]
        if len(boot_mr) else [None, None],
        "note": "model path ratio only; a near-zero query response is not clinical mediation evidence",
    }
    abs_ie = np.abs(ie)
    abs_de = np.abs(de)
    cohort_denominator = float((abs_ie + abs_de).sum())
    summary["path"]["MR_abs_cohort"] = (
        float(abs_ie.sum() / cohort_denominator) if cohort_denominator > 1e-8 else None)
    boot_abs_den = (abs_ie[picks] + abs_de[picks]).sum(axis=1)
    boot_abs = np.divide(abs_ie[picks].sum(axis=1), boot_abs_den,
                         out=np.full_like(boot_abs_den, np.nan, dtype=float),
                         where=boot_abs_den > 1e-8)
    finite_abs = boot_abs[np.isfinite(boot_abs)]
    summary["path"]["MR_abs_cohort_ci95"] = (
        [float(v) for v in np.quantile(finite_abs, [0.025, 0.975])]
        if len(finite_abs) else [None, None])
    summary["path"]["MR_abs_individual_mean"] = summary["path"].pop("MR_abs_individual")
    summary["path"]["TE_equals_DE_plus_IE_max_error"] = float(np.max(np.abs(te - de - ie)))
    summary["path"]["TE_equals_reverse_DE_plus_IE_max_error"] = float(np.max(np.abs(
        te - np.asarray([r["DE_reverse_model"] + r["IE_reverse_model"]
                         for r in positive_rows]))))
    sensitivity = []
    for shift in range(donor_sensitivity_shifts):
        shifted = {name: float(np.mean([r[f"baseline_shift_{shift}_{name}"]
                                       for r in positive_rows]))
                   for name in ("TE", "DE", "IE")}
        shifted["shift"] = shift
        shifted["MR_raw"] = (shifted["IE"] / shifted["TE"]
                             if abs(shifted["TE"]) >= 0.01 else None)
        sensitivity.append(shifted)
    summary["baseline_sensitivity"] = {
        "scope": "fixed_set_of_negative_training_donors_with_cyclic_assignment_shifts",
        "n_donors": len(baseline_ids),
        "shifts": sensitivity,
        "IE_mean_range": [min(item["IE"] for item in sensitivity),
                          max(item["IE"] for item in sensitivity)],
        "DE_mean_range": [min(item["DE"] for item in sensitivity),
                          max(item["DE"] for item in sensitivity)],
        "note": "descriptive donor sensitivity; not a confidence interval over all possible donor samples",
    }
    module_abs = {}
    for name in module_names:
        vals = [r[f"IIE_{name}"] for r in rows]
        valid_vals = np.asarray(vals, dtype=np.float64)
        valid_vals = valid_vals[np.isfinite(valid_vals)]
        stat = _mean_ci(vals, seed, n_bootstrap)
        stat["mean_absolute_response"] = (
            float(np.mean(np.abs(valid_vals))) if valid_vals.size else None)
        stat["fraction_patients_with_observed_intervention"] = float(
            np.mean([r[f"IIE_{name}_eligible_cells"] > 0 for r in rows]))
        stat["mean_eligible_cells"] = float(np.mean(
            [r[f"IIE_{name}_eligible_cells"] for r in rows]))
        local_vals = [r[f"IIE_same_label_nearest_{name}"] for r in rows]
        stat["same_label_nearest_response"] = _mean_ci(local_vals, seed, n_bootstrap)
        local_finite = np.asarray(local_vals, dtype=np.float64)
        local_finite = local_finite[np.isfinite(local_finite)]
        stat["same_label_nearest_mean_absolute_response"] = (
            float(np.mean(np.abs(local_finite))) if local_finite.size else None)
        stat["by_label"] = {
            str(label): _mean_ci([r[f"IIE_{name}"] for r in rows if r["label"] == label],
                                  seed, n_bootstrap)
            for label in (0, 1)
        }
        summary["modules"][name] = stat
        module_abs[name] = stat["mean_absolute_response"]
    total_abs = sum(value for value in module_abs.values() if value is not None)
    common_rows = [r for r in rows if all(np.isfinite(r[f"IIE_{name}"])
                                           for name in module_names)]
    common_abs = {
        name: float(np.mean([abs(r[f"IIE_{name}"]) for r in common_rows]))
        for name in module_names
    } if common_rows else {}
    common_total = sum(common_abs.values())
    summary["module_comparison_common_patient_n"] = len(common_rows)
    for name in module_names:
        summary["modules"][name]["relative_model_response_percent"] = (
            float(100 * module_abs[name] / total_abs)
            if total_abs > 0 and module_abs[name] is not None else None)
        summary["modules"][name]["relative_model_response_percent_common_patients"] = (
            float(100 * common_abs[name] / common_total)
            if common_rows and common_total > 0 else None)
    for j, name in enumerate(feature_names):
        if include_features:
            summary["features"][name] = _mean_ci(
                [r[f"feature_{j}_IIE"] for r in rows], seed, n_bootstrap)
            summary["features"][name]["fraction_patients_with_observed_intervention"] = float(
                np.mean([r[f"feature_{j}_IIE_eligible_cells"] > 0 for r in rows]))
            summary["features"][name]["by_label"] = {
                str(label): _mean_ci(
                    [r[f"feature_{j}_IIE"] for r in rows if r["label"] == label],
                    seed, n_bootstrap)
                for label in (0, 1)
            }
        if include_hte:
            plus_stat = _mean_ci([r[f"feature_{j}_plus_response"] for r in rows], seed, n_bootstrap)
            minus_stat = _mean_ci([r[f"feature_{j}_minus_response"] for r in rows], seed, n_bootstrap)
            paired_responses = np.asarray([
                [r[f"feature_{j}_plus_response"], r[f"feature_{j}_minus_response"]]
                for r in rows], dtype=np.float64)
            paired_responses = paired_responses[np.isfinite(paired_responses).all(axis=1)]
            symmetric_responses = (paired_responses[:, 0] + paired_responses[:, 1]) / 2
            plus_clip_values = np.asarray(
                [r[f"feature_{j}_plus_clip_fraction"] for r in rows], dtype=np.float64)
            minus_clip_values = np.asarray(
                [r[f"feature_{j}_minus_clip_fraction"] for r in rows], dtype=np.float64)
            plus_clip = (float(np.nanmean(plus_clip_values))
                         if np.isfinite(plus_clip_values).any() else None)
            minus_clip = (float(np.nanmean(minus_clip_values))
                          if np.isfinite(minus_clip_values).any() else None)
            def excludes_zero(stat):
                lo, hi = stat["ci95"]
                return lo is not None and (lo > 0 or hi < 0)
            coherent = (excludes_zero(plus_stat) and excludes_zero(minus_stat)
                        and plus_stat["mean"] * minus_stat["mean"] > 0
                        and plus_clip is not None and minus_clip is not None
                        and max(plus_clip, minus_clip) <= 0.10)
            summary["hte"][name] = {
                "plus_half_sd": plus_stat, "minus_half_sd": minus_stat,
                "patient_symmetric_response": _mean_ci(
                    symmetric_responses, seed, n_bootstrap),
                "patient_direction_consistency_fraction": (
                    float(np.mean(np.sign(paired_responses[:, 0])
                                  == np.sign(paired_responses[:, 1])))
                    if len(paired_responses) else None),
                "plus_clip_fraction": plus_clip, "minus_clip_fraction": minus_clip,
                "fraction_patients_with_observed_perturbation": float(np.mean(
                    [r[f"feature_{j}_perturb_eligible_cells"] > 0 for r in rows])),
                "by_label": {
                    str(label): {
                        "plus_half_sd": _mean_ci(
                            [r[f"feature_{j}_plus_response"] for r in rows if r["label"] == label],
                            seed, n_bootstrap),
                        "minus_half_sd": _mean_ci(
                            [r[f"feature_{j}_minus_response"] for r in rows if r["label"] == label],
                            seed, n_bootstrap),
                    }
                    for label in (0, 1)
                },
                "two_sided_consistent_response": bool(coherent),
                "exploratory_only": True,
                "fdr_q_values_available": False,
                "criterion": "both patient-bootstrap 95% CIs exclude zero with same sign and clip fraction <= 0.10; exploratory across 58 features",
            }
    if include_hte:
        tests = []
        for j, name in enumerate(feature_names):
            for direction in ("plus", "minus"):
                field = f"feature_{j}_{direction}_response"
                p_value = _sign_flip_mean_p(
                    [r[field] for r in rows], seed + j * 13 + (0 if direction == "plus" else 1),
                    n_permutations=max(1000, n_bootstrap),
                )
                tests.append((name, direction, p_value))
        q_values = _benjamini_hochberg([item[2] for item in tests])
        for (name, direction, p_value), q_value in zip(tests, q_values):
            summary["hte"][name][f"{direction}_sign_flip_p"] = p_value
            summary["hte"][name][f"{direction}_fdr_q"] = float(q_value)
        for name in feature_names:
            item = summary["hte"][name]
            item["two_sided_consistent_fdr"] = bool(
                item["two_sided_consistent_response"]
                and item["plus_fdr_q"] < 0.05 and item["minus_fdr_q"] < 0.05
            )
            item["fdr_q_values_available"] = True
            item["fdr_scope"] = "116 directional tests across 58 features"
    if stability_banks:
        stability = _reference_stability_summary(
            rows, module_names, stability_banks, reference_lambda,
            n_bootstrap, seed)
        with open(os.path.join(output_dir, "reference_stability.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(stability, handle, ensure_ascii=False, indent=2, allow_nan=False)
        summary["reference_stability_file"] = "reference_stability.json"
    if random_control:
        random_summary = _random_control_summary(
            rows, module_names, n_bootstrap, seed, random_match_candidates)
        with open(os.path.join(output_dir, "random_intervention_control.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(random_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        summary["random_intervention_control_file"] = "random_intervention_control.json"
    with open(os.path.join(output_dir, "model_explanation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return summary
