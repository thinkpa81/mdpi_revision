from __future__ import annotations
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "revision/tables"
RESULTS = ROOT / "revision/results"
SOURCE_HASHES = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(name):
    path = TABLES / name
    SOURCE_HASHES[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def normalize_config(value):
    if pd.isna(value):
        return "null"
    return json.dumps(json.loads(str(value)), sort_keys=True)


def regularization_relation(algo, config, point_config):
    c = json.loads(config)
    p = json.loads(point_config)
    if c is None:
        return "benchmark_no_configuration"
    if c == p:
        return "same_configuration"
    if algo == "ridge":
        return "stronger_regularization" if float(c) < float(p) else "weaker_regularization"
    assert c["learning_rate"] == p["learning_rate"]
    return (
        "lower_tree_complexity"
        if c["max_leaf_nodes"] < p["max_leaf_nodes"]
        else "higher_tree_complexity"
    )


def aggregate_public():
    RESULTS.mkdir(parents=True, exist_ok=True)
    policies = source("selected_budget_policy_results.csv")
    assert len(policies) == 248
    assert not policies.duplicated(["algorithm", "history_available", "budget", "domain"]).any()
    inference = source("fixedfit_paired_harm_benefit_inference.csv")
    inference = inference[inference.family.eq("original_all_budgets_two_references")]
    for r in policies.itertuples(index=False):
        for ref_name in ("benchmark", "full"):
            ref = r.benchmark if ref_name == "benchmark" else r.full_model
            expected_low = getattr(r, "lower95_vs_" + ref_name)
            expected_high = getattr(r, "upper95_vs_" + ref_name)
            if r.chosen_model == ref:
                assert expected_low == expected_high == 0
                continue
            v = inference[
                (inference.algorithm == r.algorithm)
                & (inference.history_available == r.history_available)
                & (inference.domain == r.domain)
                & (inference.model == r.chosen_model)
                & (inference.reference == ref)
            ]
            assert len(v) == 1
            v = v.iloc[0]
            assert abs(expected_low - v.simultaneous_lower95_one_sided) < 1e-12
            assert abs(expected_high - v.simultaneous_upper95_one_sided) < 1e-12
    policy_cols = [
        "algorithm",
        "history_available",
        "budget",
        "domain",
        "n",
        "chosen_model",
        "benchmark",
        "full_model",
        "log_loss",
        "delta_vs_benchmark",
        "lower95_vs_benchmark",
        "upper95_vs_benchmark",
        "delta_vs_full",
        "lower95_vs_full",
        "upper95_vs_full",
        "harm_vs_benchmark",
        "benefit_vs_benchmark",
    ]
    policies[policy_cols].sort_values(
        ["algorithm", "history_available", "domain", "budget"]
    ).to_csv(RESULTS / "submission_budget_bounds.csv", index=False)
    policy_summary = []
    for key, g in policies.groupby(["domain", "algorithm", "history_available"]):
        assert len(g) == 31
        policy_summary.append(
            dict(zip(["domain", "algorithm", "history_available"], key))
            | {
                "budgets": len(g),
                "unique_chosen_models": int(g.chosen_model.nunique()),
                "harm_budgets": int((g.lower95_vs_benchmark > 0).sum()),
                "benefit_budgets": int((g.upper95_vs_benchmark < 0).sum()),
                "inconclusive_or_identical_budgets": int(
                    ((g.lower95_vs_benchmark <= 0) & (g.upper95_vs_benchmark >= 0)).sum()
                ),
                "inference": "separate one-sided 95% simultaneous bounds within each original algorithm/history/domain family including both references",
            }
        )
    pd.DataFrame(policy_summary).to_csv(
        RESULTS / "submission_budget_harm_summary.csv", index=False
    )
    family_frames = []
    for name, label, size in [
        ("model_sensitivity_full_model_intervals.csv", "expanded_four_learners", 8),
        ("round2_boosting_full_model_intervals.csv", "additional_three_learners", 6),
    ]:
        frame = source(name)
        for key, g in frame.groupby(["population", "evaluation_group"]):
            assert len(g) == size
            family_frames.append(
                {
                    "model_family": label,
                    "training_population": key[0],
                    "evaluation_group": key[1],
                    "n": int(g.n.iloc[0]),
                    "contrasts": len(g),
                    "supported_benefit": int((g.simultaneous_high < 0).sum()),
                    "supported_harm": int((g.simultaneous_low > 0).sum()),
                    "inconclusive": int(
                        ((g.simultaneous_low <= 0) & (g.simultaneous_high >= 0)).sum()
                    ),
                    "family_size": size,
                    "interval": "two-sided 95% simultaneous fixed-prediction interval",
                }
            )
    original = source("full_models_vs_benchmarks.csv")
    for domain, g in original.groupby("domain"):
        assert len(g) == 4
        family_frames.append(
            {
                "model_family": "original_two_learners",
                "training_population": "restricted",
                "evaluation_group": "all" if domain == "whole" else "person_disjoint",
                "n": int(g.n.iloc[0]),
                "contrasts": len(g),
                "supported_benefit": int((g.simultaneous_upper95_two_sided < 0).sum()),
                "supported_harm": int((g.simultaneous_lower95_two_sided > 0).sum()),
                "inconclusive": int(
                    (
                        (g.simultaneous_lower95_two_sided <= 0)
                        & (g.simultaneous_upper95_two_sided >= 0)
                    ).sum()
                ),
                "family_size": 4,
                "interval": "two-sided 95% simultaneous fixed-prediction interval",
            }
        )
    model_summary = pd.DataFrame(family_frames)
    model_summary.to_csv(RESULTS / "submission_model_family_summary.csv", index=False)
    b = source("round2_reselection_bootstrap_replicates.csv")
    p = source("round2_reselection_point_estimates.csv")
    b["config_key"] = b.selected_config.map(normalize_config)
    p["config_key"] = p.selected_config.map(normalize_config)
    assert len(b) == 8000 and b.replicate.nunique() == 1000
    unique = b.drop_duplicates("replicate")
    for col in ["n_development", "unique_development", "n_evaluation", "unique_evaluation"]:
        assert b.groupby("replicate")[col].nunique().eq(1).all()
    unique_summary = []
    for split in ["development", "evaluation"]:
        share = unique["unique_" + split] / unique["n_" + split]
        unique_summary.append(
            {
                "split": split,
                "resamples": len(unique),
                "mean_rows": float(unique["n_" + split].mean()),
                "mean_unique_participants": float(unique["unique_" + split].mean()),
                "mean_unique_fraction": float(share.mean()),
                "fraction_025": float(share.quantile(0.025)),
                "fraction_975": float(share.quantile(0.975)),
                "fraction_min": float(share.min()),
                "fraction_max": float(share.max()),
            }
        )
    pd.DataFrame(unique_summary).to_csv(
        RESULTS / "submission_bootstrap_unique_participants.csv", index=False
    )
    cfg_rows = []
    relation_rows = []
    for key, g in b.groupby(["algorithm", "history_available", "budget"]):
        point = p[
            (p.algorithm == key[0]) & (p.history_available == key[1]) & (p.budget == key[2])
        ].iloc[0]
        meta = dict(zip(["algorithm", "history_available", "budget"], key))
        meta.update(
            point_subset=point.selected_subset,
            point_configuration=point.config_key,
            point_delta_vs_benchmark=float(point.delta_vs_benchmark),
            point_delta_vs_full=float(point.delta_vs_full),
        )
        g = g.copy()
        g["regularization_relation"] = g.config_key.map(
            lambda c: regularization_relation(key[0], c, point.config_key)
        )
        for columns, destination in [
            (["config_key"], cfg_rows),
            (["regularization_relation"], relation_rows),
        ]:
            for group_key, h in g.groupby(columns):
                record = meta | {
                    columns[0]: group_key[0],
                    "replicates": len(h),
                    "frequency": len(h) / len(g),
                    "mean_delta_vs_benchmark": float(h.delta_vs_benchmark.mean()),
                    "mean_delta_vs_full": float(h.delta_vs_full.mean()),
                    "delta_vs_benchmark_025": float(h.delta_vs_benchmark.quantile(0.025)),
                    "delta_vs_benchmark_975": float(h.delta_vs_benchmark.quantile(0.975)),
                    "point_portfolio_frequency_within_configuration": float(
                        h.selected_subset.eq(point.selected_subset).mean()
                    ),
                    "mean_unique_development_fraction": float(
                        (h.unique_development / h.n_development).mean()
                    ),
                }
                if columns[0] == "config_key":
                    record["regularization_relation"] = regularization_relation(
                        key[0], group_key[0], point.config_key
                    )
                destination.append(record)
    pd.DataFrame(cfg_rows).to_csv(
        RESULTS / "submission_selection_configuration_summary.csv", index=False
    )
    relation_table = pd.DataFrame(relation_rows)
    relation_table.to_csv(RESULTS / "submission_selection_regularization_summary.csv", index=False)
    compact = []
    for key, g in relation_table[relation_table.budget.eq(4)].groupby(
        ["algorithm", "history_available"]
    ):
        same = g[g.regularization_relation.eq("same_configuration")].iloc[0]
        stronger = g[
            g.regularization_relation.isin(["stronger_regularization", "lower_tree_complexity"])
        ]
        weaker = g[
            g.regularization_relation.isin(["weaker_regularization", "higher_tree_complexity"])
        ]
        assert len(stronger) == 1
        stronger = stronger.iloc[0]
        compact.append(
            {
                "algorithm": key[0],
                "history_available": key[1],
                "budget": 4,
                "point_configuration": same.point_configuration,
                "point_delta_vs_benchmark": same.point_delta_vs_benchmark,
                "same_configuration_replicates": int(same.replicates),
                "same_configuration_frequency": same.frequency,
                "same_configuration_mean_delta": same.mean_delta_vs_benchmark,
                "stronger_or_lower_complexity_replicates": int(stronger.replicates),
                "stronger_or_lower_complexity_frequency": stronger.frequency,
                "stronger_or_lower_complexity_mean_delta": stronger.mean_delta_vs_benchmark,
                "weaker_or_higher_complexity_replicates": int(weaker.replicates.sum()),
                "weaker_or_higher_complexity_frequency": float(weaker.frequency.sum()),
                "weaker_or_higher_complexity_mean_delta": (
                    float(weaker.mean_delta_vs_benchmark.iloc[0]) if len(weaker) else None
                ),
                "interpretation": "descriptive conditional associations; not causal effects or new confidence intervals",
            }
        )
    pd.DataFrame(compact).to_csv(
        RESULTS / "submission_budget4_configuration_summary.csv", index=False
    )
    expanded_restricted = model_summary[
        (model_summary.training_population == "restricted")
        & (model_summary.evaluation_group == "all")
        & (model_summary.model_family != "original_two_learners")
    ]
    hesitant = model_summary[
        (model_summary.training_population == "restricted")
        & (model_summary.evaluation_group == "all")
        | (model_summary.training_population == "full")
        & (model_summary.evaluation_group == "hesitant")
    ]
    grouped = {}
    for group in [
        "all",
        "person_disjoint",
        "other",
        "low_or_no_hesitation",
        "no_service_experience",
    ]:
        g = model_summary[
            (model_summary.training_population == "full")
            & (model_summary.evaluation_group == group)
        ]
        grouped[group] = {
            k: int(g[k].sum()) for k in ["contrasts", "supported_benefit", "supported_harm"]
        }
    claims = {
        "restricted_expanded_and_added": {
            k: int(expanded_restricted[k].sum())
            for k in ["contrasts", "supported_benefit", "supported_harm"]
        },
        "hesitant_evaluation_all_model_sets": {
            k: int(hesitant[k].sum()) for k in ["contrasts", "supported_benefit", "supported_harm"]
        },
        "hesitant_32_denominator": "4 original restricted-training models + 14 expanded/added restricted-training models + 14 expanded/added full-training models; not 2 x 7 alone",
        "full_training_groups": grouped,
        "budget_policies": {
            "policy_rows": len(policies),
            "supported_benefit": int(policies.benefit_vs_benchmark.sum()),
            "supported_harm": int(policies.harm_vs_benchmark.sum()),
            "unit": "budget policy rows; repeated selected models are not independent tests",
        },
        "group_definitions": {
            "hesitant": "privacy-related purchase hesitation codes 1-2; n=1520",
            "low_or_no_hesitation": "codes 3-4; n=1995",
            "no_service_experience": "code 9; n=226",
            "other": "codes 3,4,9; n=2221; includes participants without service experience",
        },
        "interpretation": [
            "Counts summarize separate simultaneous families; there is no joint 95% guarantee across all 32 or 248 rows.",
            "No supported benefit is not a demonstration of equivalence or zero predictive benefit.",
            "A unique participant fraction near 0.632 is a routine consequence of sampling with replacement, not by itself evidence of invalidity.",
            "Configuration shifts accompany the resampling distribution and lower average loss differences; conditional summaries do not identify a causal contribution to bootstrap bias.",
            "Smaller ridge C means stronger penalization; fewer HGB leaves means lower tree complexity.",
            "The broad other group must not be labeled unqualified non-hesitant because it includes code 9.",
        ],
    }
    return claims


def overlap_refit(development, evaluation):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import original_model_engine as eng

    dev = pd.read_csv(development, dtype={"id": str})
    ev = pd.read_csv(evaluation, dtype={"id": str})
    assert dev.id.is_unique and ev.id.is_unique
    assert len(dev) == 1565 and len(ev) == 1520
    SOURCE_HASHES["private_development_input"] = digest(development)
    SOURCE_HASHES["private_evaluation_input"] = digest(evaluation)
    cfg_path = ROOT / "revision/audit/original_frozen_configs.json"
    configs = {r["algorithm"] + "::" + r["subset"]: r for r in json.loads(cfg_path.read_text())}
    SOURCE_HASHES[str(cfg_path.relative_to(ROOT))] = digest(cfg_path)
    y = ev.outcome_any_use.to_numpy(int)
    baseline = {
        "P0": eng.p0(dev, ev),
        "NONE": np.full(len(ev), (dev.outcome_any_use.sum() + 1) / (len(dev) + 2)),
    }
    meta = []
    differences = []
    predictions = []
    with threadpool_limits(1):
        for algo in ["ridge", "hgb"]:
            for history in [False, True]:
                subset = ("H_" if history else "") + "D_B_M_P_C_A_K"
                row = configs[algo + "::" + subset]
                prediction, _ = eng.fit_predict(dev, ev, row["blocks"], algo, row["best_config"])
                benchmark = "P0" if history else "NONE"
                ll = eng.loss_vector(y, prediction)
                base_ll = eng.loss_vector(y, baseline[benchmark])
                predictions.append((ll, base_ll))
                differences.append(ll - base_ll)
                meta.append(
                    {
                        "algorithm": algo,
                        "history_available": history,
                        "full_subset": subset,
                        "benchmark": benchmark,
                        "configuration": json.dumps(row["best_config"], sort_keys=True),
                    }
                )
    D = np.column_stack(differences)
    shared = ev.id.isin(dev.id).to_numpy()
    assert int(shared.sum()) == 641
    masks = {"whole": np.ones(len(ev), bool), "overlap": shared, "disjoint": ~shared}
    rng = np.random.default_rng(20261020)
    weights = rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)), size=5000)
    rows = []
    draws = {}
    means = {}
    reference = source("full_models_vs_benchmarks.csv")
    max_existing_difference = 0.0
    with threadpool_limits(1):
        for group, mask in masks.items():
            w = weights[:, mask].astype(float)
            w /= w.sum(axis=1)[:, None]
            means[group] = D[mask].mean(axis=0)
            draws[group] = w @ D[mask]
            centered = draws[group] - means[group]
            critical = float(np.quantile(np.abs(centered).max(axis=1), 0.95))
            for j, m in enumerate(meta):
                row = m | {
                    "evaluation_group": group,
                    "n": int(mask.sum()),
                    "model_log_loss": float(predictions[j][0][mask].mean()),
                    "benchmark_log_loss": float(predictions[j][1][mask].mean()),
                    "delta_log_loss": float(means[group][j]),
                    "simultaneous_low": float(means[group][j] - critical),
                    "simultaneous_high": float(means[group][j] + critical),
                    "family_size": 4,
                    "bootstrap_repetitions": 5000,
                    "inference": "two-sided 95% simultaneous fixed-prediction interval, separate four-model family per group",
                }
                rows.append(row)
                if group in ["whole", "disjoint"]:
                    r = reference[
                        (reference.domain == group)
                        & (reference.algorithm == m["algorithm"])
                        & (reference.history_available == m["history_available"])
                    ].iloc[0]
                    diffs = [
                        abs(row["delta_log_loss"] - r.delta_full_vs_benchmark),
                        abs(row["simultaneous_low"] - r.simultaneous_lower95_two_sided),
                        abs(row["simultaneous_high"] - r.simultaneous_upper95_two_sided),
                    ]
                    max_existing_difference = max(max_existing_difference, *diffs)
    assert max_existing_difference < 1e-10, max_existing_difference
    pd.DataFrame(rows).to_csv(RESULTS / "submission_overlap_full_models.csv", index=False)
    difference = means["overlap"] - means["disjoint"]
    bootstrap_difference = draws["overlap"] - draws["disjoint"]
    critical = float(np.quantile(np.abs(bootstrap_difference - difference).max(axis=1), 0.95))
    pd.DataFrame(
        [
            m
            | {
                "contrast": "shared_minus_disjoint_loss_difference",
                "delta": float(difference[j]),
                "simultaneous_low": float(difference[j] - critical),
                "simultaneous_high": float(difference[j] + critical),
                "family_size": 4,
                "bootstrap_repetitions": 5000,
                "interpretation": "conditional predictive heterogeneity; not a causal overlap effect",
            }
            for j, m in enumerate(meta)
        ]
    ).to_csv(RESULTS / "submission_overlap_heterogeneity.csv", index=False)
    return {
        "status": "passed",
        "fresh_full_model_fits": 4,
        "n_development": len(dev),
        "n_evaluation": len(ev),
        "shared_n": int(shared.sum()),
        "shared_fraction": float(shared.mean()),
        "disjoint_n": int((~shared).sum()),
        "bootstrap_repetitions": 5000,
        "max_absolute_difference_existing_whole_disjoint_results": max_existing_difference,
        "fixed_predictions": True,
        "refit_in_each_bootstrap": False,
        "full_panel_overlap_fraction_from_existing_group_counts": (3741 - 408) / 3741,
        "full_panel_overlap_count_source": "n=3741 all and n=408 person_disjoint in model-family interval tables",
        "inference_scope": "separate four-model family within each group; a separate four-contrast family for shared-minus-disjoint heterogeneity",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize policy harm, model-family contrasts, and resampling diagnostics.\n\nThe stored estimates are preserved. Conditional configuration summaries describe\nresamples selected into each configuration; they are not causal effects of tuning.\nWith prepared development/evaluation CSVs, refit the four original full models and\ncompare shared and person-disjoint evaluation groups using fixed-fit inference.\nNo participant rows or individual predictions are written to the public outputs.\n"
    )
    parser.add_argument("--development", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if bool(args.development) != bool(args.evaluation):
        parser.error("Provide both --development and --evaluation or neither.")
    claims = aggregate_public()
    if args.development:
        claims["fresh_overlap_analysis"] = overlap_refit(args.development, args.evaluation)
    else:
        claims["fresh_overlap_analysis"] = {
            "status": "not_run",
            "reason": "prepared private inputs not supplied",
        }
    claims["source_sha256"] = SOURCE_HASHES
    claims["script_sha256"] = digest(__file__)
    claims["status"] = "passed"
    dump(RESULTS / "submission_numeric_claims.json", claims)
    print(
        json.dumps(
            {
                k: claims[k]
                for k in [
                    "status",
                    "restricted_expanded_and_added",
                    "hesitant_evaluation_all_model_sets",
                    "full_training_groups",
                    "budget_policies",
                    "fresh_overlap_analysis",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
