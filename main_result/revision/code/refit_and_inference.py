import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, hashlib, argparse, time, platform, shutil
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "original_results/main_result/mdpi-main/01_required_prior_analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import original_model_engine as eng

SEED = 20261020
B = 5000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(
        json.dumps(
            x,
            indent=2,
            ensure_ascii=False,
            default=lambda v: v.item() if hasattr(v, "item") else str(v),
        )
    )


def fit_one(r, dev, eva):
    p, n = eng.fit_predict(dev, eva, r["blocks"], r["algorithm"], r["best_config"])
    k = r["algorithm"] + "::" + r["subset"]
    return (k, p, n)


def regenerate(dev, eva):
    cfg = ROOT / "revision/audit/original_frozen_configs.json"
    if not cfg.exists():
        frozen = json.loads(
            (ORIGINAL / "analysis_outputs/frozen_development_selection.json").read_text()
        )
        compact = [
            {
                k: r[k]
                for k in (
                    "algorithm",
                    "subset",
                    "blocks",
                    "has_history",
                    "best_config",
                    "additional_items",
                    "raw_items",
                    "oof_log_loss",
                )
            }
            for r in frozen["all_models"]
        ]
        dump(cfg, compact)
    rows = json.loads(cfg.read_text())
    start = time.time()
    results = Parallel(n_jobs=3, verbose=5)((delayed(fit_one)(r, dev, eva) for r in rows))
    pred = {k: p for k, p, n in results}
    pred["P0"] = eng.p0(dev, eva)
    pred["prevalence"] = pred["ridge::NONE"]
    np.savez_compressed(ROOT / "private_data/refit_all_predictions.npz", **pred)
    dump(
        ROOT / "private_data/refit_models.json",
        {
            "row_order": "private_data/refit_prediction_ids.csv",
            "models": rows,
            "prediction_keys": list(pred),
        },
    )
    eva[["id"]].to_csv(ROOT / "private_data/refit_prediction_ids.csv", index=False)
    metrics = []
    for k, p in pred.items():
        metrics.append({"model": k, **eng.metrics(eva.outcome_any_use.values, p)})
    m = pd.DataFrame(metrics)
    m.to_csv(ROOT / "revision/tables/refit_all_model_metrics.csv", index=False)
    reference = ROOT / "revision/audit/original_model_metric_reference.csv"
    if not reference.exists():
        shutil.copy2(ORIGINAL / "analysis_outputs/evaluation_all_subsets.csv", reference)
    old = pd.read_csv(reference)
    compared = m.merge(old, on="model", suffixes=("_refit", "_original"))
    for metric in ["log_loss", "brier", "roc_auc", "average_precision"]:
        compared[metric + "_difference"] = (
            compared[metric + "_refit"] - compared[metric + "_original"]
        )
    compared[["model"] + [c for c in compared if c.endswith("_difference")]].to_csv(
        ROOT / "revision/audit/refit_original_metric_differences.csv", index=False
    )
    report = {
        "models_refit": len(rows),
        "predictions": len(pred),
        "development_n": len(dev),
        "development_events": int(dev.outcome_any_use.sum()),
        "evaluation_n": len(eva),
        "evaluation_events": int(eva.outcome_any_use.sum()),
        "development_sha256": sha(ROOT / "private_data/development.csv"),
        "evaluation_sha256": sha(ROOT / "private_data/evaluation.csv"),
        "frozen_config_sha256": sha(cfg),
        "model_engine_sha256": sha(Path(eng.__file__)),
        "elapsed_seconds": time.time() - start,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "max_abs_metric_differences": {
            c: float(compared[c].abs().max()) for c in compared if c.endswith("_difference")
        },
        "scope": "new model fits from current authorized raw data; original frozen subset and hyperparameter choices retained",
    }
    dump(ROOT / "revision/audit/refit_reproduction_report.json", report)
    print(json.dumps(report), flush=True)
    return pred


def validate_original_bounds(pred, dev, eva):
    y = eva.outcome_any_use.to_numpy(int)
    n = len(y)
    reference = ROOT / "revision/audit/original_budget_bounds_reference.csv"
    if not reference.exists():
        shutil.copy2(
            ORIGINAL / "revision_20260907/outputs/budget_policy_loss_boundaries.csv", reference
        )
    polpath = ROOT / "revision/audit/original_budget_policies.csv"
    if not polpath.exists():
        shutil.copy2(
            ORIGINAL / "revision_20260907/outputs/development_budget_policies.csv", polpath
        )
    old = pd.read_csv(reference)
    pol = pd.read_csv(polpath)
    weights = np.random.default_rng(20260909).multinomial(n, np.ones(n) / n, size=2000)
    disjoint = ~eva.id.astype(str).isin(set(dev.id.astype(str))).to_numpy()
    rows = []
    for algo in ("ridge", "hgb"):
        for avail in (False, True):
            fam = pol[(pol.algorithm == algo) & (pol.history_available == avail)]
            base = "P0" if avail else algo + "::NONE"
            full = algo + "::" + "_".join((["H"] if avail else []) + eng.OPTIONAL)
            contrasts = list(
                dict.fromkeys(((k, r) for k in fam.chosen_model for r in (full, base) if k != r))
            )
            keys = list(dict.fromkeys([x for pair in contrasts for x in pair]))
            keyidx = {k: j for j, k in enumerate(keys)}
            for domain, mask in [("whole", np.ones(n, bool)), ("disjoint", disjoint)]:
                losses = np.column_stack([eng.loss_vector(y[mask], pred[k][mask]) for k in keys])
                means = losses.mean(0)
                w = weights[:, mask].astype(float)
                w /= w.sum(1)[:, None]
                boot = w @ losses
                d = np.array([means[keyidx[k]] - means[keyidx[r]] for k, r in contrasts])
                db = np.column_stack(
                    [boot[:, keyidx[k]] - boot[:, keyidx[r]] for k, r in contrasts]
                )
                q = np.quantile(np.max(d - db, 1), 0.95)
                for _, r in fam.iterrows():
                    k = r.chosen_model
                    match = old[(old.policy_id == r.policy_id) & (old.domain == domain)].iloc[0]
                    for ref, kind in [(base, "benchmark"), (full, "full")]:
                        upper = 0 if k == ref else means[keyidx[k]] - means[keyidx[ref]] + q
                        rows.append(
                            {
                                "policy_id": r.policy_id,
                                "domain": domain,
                                "reference": kind,
                                "recomputed_upper95": upper,
                                "original_upper95": match["upper95_vs_" + kind],
                                "difference": upper - match["upper95_vs_" + kind],
                                "unique_contrasts": len(contrasts),
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "revision/audit/original_bound_reproduction.csv", index=False)
    dump(
        ROOT / "revision/audit/original_bound_reproduction.json",
        {
            "bootstrap_replicates": 2000,
            "seed": 20260909,
            "rows": len(df),
            "max_absolute_difference": float(df.difference.abs().max()),
            "method": "original unstudentized max centered bootstrap upper bound and original family retained exactly",
        },
    )
    print("ORIGINAL BOUND CHECK", float(df.difference.abs().max()), flush=True)


def inference(pred, dev, eva):
    polfile = ROOT / "revision/audit/original_budget_policies.csv"
    if not polfile.exists():
        shutil.copy2(
            ORIGINAL / "revision_20260907/outputs/development_budget_policies.csv", polfile
        )
    pol = pd.read_csv(polfile)
    allrows = []
    familyrows = []
    fullrows = []
    metricrows = []
    weightedrows = []
    rng = np.random.default_rng(SEED)
    n = len(eva)
    y = eva.outcome_any_use.to_numpy(int)
    weights = rng.multinomial(n, np.repeat(1 / n, n), size=B).astype(float)
    disjoint = ~eva.id.astype(str).isin(set(dev.id.astype(str))).to_numpy()
    np.savez_compressed(
        ROOT / "private_data/fixedfit_bootstrap_weights.npz", weights=weights.astype(np.uint8)
    )
    masks = {"whole": np.ones(n, bool), "disjoint": disjoint}
    for domain, mask in masks.items():
        w = weights[:, mask]
        w /= w.sum(1)[:, None]
        yy = y[mask]
        keys = list(pred)
        losses = np.column_stack([eng.loss_vector(yy, pred[k][mask]) for k in keys])
        means = losses.mean(0)
        boot = w @ losses
        keyidx = {k: j for j, k in enumerate(keys)}
        all_full = []
        for algo in ("ridge", "hgb"):
            for avail in (False, True):
                fam = pol[(pol.algorithm == algo) & (pol.history_available == avail)].copy()
                base = "P0" if avail else algo + "::NONE"
                full = algo + "::" + "_".join((["H"] if avail else []) + eng.OPTIONAL)
                orig_con = list(
                    dict.fromkeys(
                        (
                            (r.chosen_model, ref)
                            for _, r in fam.iterrows()
                            for ref in (full, base)
                            if r.chosen_model != ref
                        )
                    )
                )
                sensitivity = {
                    "original_all_budgets_two_references": orig_con,
                    "postreview_all_budgets_benchmark_only": list(
                        dict.fromkeys(
                            (
                                (r.chosen_model, base)
                                for _, r in fam.iterrows()
                                if r.chosen_model != base
                            )
                        )
                    ),
                    "postreview_budgets_0_to_4_benchmark_only": list(
                        dict.fromkeys(
                            (
                                (r.chosen_model, base)
                                for _, r in fam[fam.budget <= 4].iterrows()
                                if r.chosen_model != base
                            )
                        )
                    ),
                }
                for family, contrasts in sensitivity.items():
                    d = np.array([means[keyidx[k]] - means[keyidx[r]] for k, r in contrasts])
                    db = np.column_stack(
                        [boot[:, keyidx[k]] - boot[:, keyidx[r]] for k, r in contrasts]
                    )
                    centered = db - d
                    upper_critical = float(np.quantile(np.max(-centered, axis=1), 0.95))
                    lower_critical = float(np.quantile(np.max(centered, axis=1), 0.95))
                    critical_abs = float(np.quantile(np.max(np.abs(centered), axis=1), 0.95))
                    se = centered.std(0, ddof=1)
                    valid = se > 1e-14
                    student_center = np.divide(
                        centered, se, where=valid, out=np.zeros_like(centered)
                    )
                    student_upper = float(np.quantile(np.max(-student_center, axis=1), 0.95))
                    student_lower = float(np.quantile(np.max(student_center, axis=1), 0.95))
                    familyrows.append(
                        {
                            "algorithm": algo,
                            "history_available": avail,
                            "domain": domain,
                            "family": family,
                            "n": int(mask.sum()),
                            "bootstrap_replicates": B,
                            "unique_nonzero_contrasts": len(contrasts),
                            "upper_critical": upper_critical,
                            "lower_critical": lower_critical,
                            "absolute_two_sided_critical": critical_abs,
                            "studentized_upper_critical": student_upper,
                            "studentized_lower_critical": student_lower,
                            "scope": "conditional on frozen fitted predictions; coverage separately within family",
                        }
                    )
                    for j, (model, ref) in enumerate(contrasts):
                        harmmax = np.max(centered, axis=1)
                        benefitmax = np.max(-centered, axis=1)
                        r = {
                            "algorithm": algo,
                            "history_available": avail,
                            "domain": domain,
                            "family": family,
                            "model": model,
                            "reference": ref,
                            "n": int(mask.sum()),
                            "delta_log_loss": d[j],
                            "point_lower95_one_sided": d[j] - np.quantile(centered[:, j], 0.95),
                            "point_upper95_one_sided": d[j] - np.quantile(centered[:, j], 0.05),
                            "point_lower95_two_sided": d[j] - np.quantile(centered[:, j], 0.975),
                            "point_upper95_two_sided": d[j] - np.quantile(centered[:, j], 0.025),
                            "simultaneous_lower95_one_sided": d[j] - lower_critical,
                            "simultaneous_upper95_one_sided": d[j] + upper_critical,
                            "simultaneous_lower95_two_sided": d[j] - critical_abs,
                            "simultaneous_upper95_two_sided": d[j] + critical_abs,
                            "harm_p_adjusted_bootstrap": (1 + np.sum(harmmax >= d[j])) / (B + 1),
                            "benefit_p_adjusted_bootstrap": (1 + np.sum(benefitmax >= -d[j]))
                            / (B + 1),
                            "harm_p_pointwise_bootstrap": (1 + np.sum(centered[:, j] >= d[j]))
                            / (B + 1),
                            "benefit_p_pointwise_bootstrap": (1 + np.sum(-centered[:, j] >= -d[j]))
                            / (B + 1),
                            "studentized_lower95": d[j] - student_lower * se[j],
                            "studentized_upper95": d[j] + student_upper * se[j],
                            "approx_MDE80_pointwise": (norm.ppf(0.95) + norm.ppf(0.8)) * se[j],
                            "approx_MDE80_simultaneous": upper_critical + norm.ppf(0.8) * se[j],
                            "harm_detected": bool(d[j] - lower_critical > 0),
                            "benefit_detected": bool(d[j] + upper_critical < 0),
                            "direction": "positive delta is model harm; negative delta is model benefit",
                            "analysis_status": (
                                "original comparison family"
                                if family.startswith("original")
                                else "post-review exploratory sensitivity"
                            ),
                        }
                        for margin in (0, 0.0025, 0.005, 0.01):
                            r["noninferior_margin_" + str(margin)] = bool(
                                r["simultaneous_upper95_one_sided"] < margin
                            )
                        allrows.append(r)
                for k in [base, full]:
                    all_full.append((algo, avail, k, base, full))
                selected = list(
                    dict.fromkeys(
                        [base, full] + fam[fam.budget.isin([0, 1, 4, 30])].chosen_model.tolist()
                    )
                )
                for model in selected:
                    p = pred[model][mask]
                    mm = eng.metrics(yy, p)
                    np_rng = np.random.default_rng(SEED + 31)
                    auc = []
                    ll = []
                    for b in range(2000):
                        idx = np_rng.integers(0, len(yy), len(yy))
                        if len(np.unique(yy[idx])) < 2:
                            continue
                        auc.append(roc_auc_score(yy[idx], p[idx]))
                        ll.append(eng.loss_vector(yy[idx], p[idx]).mean())
                    metricrows.append(
                        {
                            "algorithm": algo,
                            "history_available": avail,
                            "domain": domain,
                            "model": model,
                            **mm,
                            "auc_lower95": np.quantile(auc, 0.025),
                            "auc_upper95": np.quantile(auc, 0.975),
                            "log_loss_lower95": np.quantile(ll, 0.025),
                            "log_loss_upper95": np.quantile(ll, 0.975),
                            "ci_replicates": len(auc),
                            "ci_method": "percentile participant bootstrap; fixed fitted predictions",
                        }
                    )
                    vm = (
                        mask
                        & np.isfinite(eva.outcome_long_weight.to_numpy())
                        & (eva.outcome_long_weight.to_numpy() > 0)
                    )
                    for weighting in (
                        "valid_weight_subset_unweighted",
                        "valid_weight_subset_longitudinal_weighted",
                    ):
                        sw = (
                            eva.loc[vm, "outcome_long_weight"].values
                            if "longitudinal_weighted" in weighting
                            else None
                        )
                        wd = eng.metrics(y[vm], pred[model][vm], sw)
                        delta = np.average(
                            eng.loss_vector(y[vm], pred[model][vm])
                            - eng.loss_vector(y[vm], pred[base][vm]),
                            weights=sw,
                        )
                        weightedrows.append(
                            {
                                "algorithm": algo,
                                "history_available": avail,
                                "domain": domain,
                                "model": model,
                                "benchmark": base,
                                "analysis": weighting,
                                **wd,
                                "delta_vs_benchmark": delta,
                                "effective_n": (
                                    float(sw.sum() ** 2 / (sw @ sw))
                                    if sw is not None
                                    else int(vm.sum())
                                ),
                            }
                        )
        direct = [(a, h, full, base) for a, h, k, base, full in all_full if k == full]
        dd = np.array([means[keyidx[full]] - means[keyidx[base]] for a, h, full, base in direct])
        db = np.column_stack(
            [boot[:, keyidx[full]] - boot[:, keyidx[base]] for a, h, full, base in direct]
        )
        c = db - dd
        cu = np.quantile(np.max(-c, 1), 0.95)
        cl = np.quantile(np.max(c, 1), 0.95)
        ca = np.quantile(np.max(abs(c), 1), 0.95)
        for j, (a, h, full, base) in enumerate(direct):
            fullrows.append(
                {
                    "algorithm": a,
                    "history_available": h,
                    "domain": domain,
                    "n": int(mask.sum()),
                    "full_model": full,
                    "benchmark": base,
                    "full_log_loss": means[keyidx[full]],
                    "benchmark_log_loss": means[keyidx[base]],
                    "P0_log_loss": means[keyidx["P0"]],
                    "NONE_log_loss": means[keyidx["ridge::NONE"]],
                    "delta_full_vs_benchmark": dd[j],
                    "point_lower95": dd[j] - np.quantile(c[:, j], 0.975),
                    "point_upper95": dd[j] - np.quantile(c[:, j], 0.025),
                    "simultaneous_lower95_one_sided": dd[j] - cl,
                    "simultaneous_upper95_one_sided": dd[j] + cu,
                    "simultaneous_lower95_two_sided": dd[j] - ca,
                    "simultaneous_upper95_two_sided": dd[j] + ca,
                    "harm_p_adjusted": (1 + sum(c.max(1) >= dd[j])) / (B + 1),
                    "reference_beats_benchmark_point_estimate": bool(dd[j] < 0),
                    "family": "post-review four full models versus respective benchmark",
                }
            )
    pd.DataFrame(allrows).to_csv(
        ROOT / "revision/tables/fixedfit_paired_harm_benefit_inference.csv", index=False
    )
    pd.DataFrame(familyrows).to_csv(
        ROOT / "revision/tables/fixedfit_inference_families.csv", index=False
    )
    pd.DataFrame(fullrows).to_csv(
        ROOT / "revision/tables/full_models_vs_benchmarks.csv", index=False
    )
    pd.DataFrame(metricrows).to_csv(
        ROOT / "revision/tables/core_models_metrics_with_auc_ci.csv", index=False
    )
    pd.DataFrame(weightedrows).to_csv(
        ROOT / "revision/tables/longitudinal_weight_sensitivity.csv", index=False
    )
    dump(
        ROOT / "revision/audit/fixedfit_inference_report.json",
        {
            "seed": SEED,
            "bootstrap_replicates": B,
            "metric_ci_replicates": 2000,
            "inference_rows": len(allrows),
            "families": len(familyrows),
            "full_model_rows": len(fullrows),
            "prediction_sha256": sha(ROOT / "private_data/refit_all_predictions.npz"),
            "sign": "LL_model minus LL_reference; positive means worse prediction",
            "limitation": "fixed-fit intervals condition on development and selection; full reselection uncertainty is a separate analysis",
            "noninferiority_margins": "0, 0.0025, 0.005, 0.01 are illustrative retrospective sensitivity thresholds, not validated business tolerances",
            "postreview_families": "exploratory sensitivity; original family retained",
            "bootstrap": "ordinary participant bootstrap on whole evaluation cohort; disjoint counts reweighted from same replicate for parity with original method",
            "MDE": "normal-approximation local effect-size planning diagnostic from observed paired SE; not achieved post-hoc power",
        },
    )
    print("INFERENCE COMPLETE", len(allrows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-only", action="store_true")
    args = ap.parse_args()
    dev = pd.read_csv(ROOT / "private_data/development.csv", dtype={"id": str})
    eva = pd.read_csv(ROOT / "private_data/evaluation.csv", dtype={"id": str})
    assert dev.id.is_unique and eva.id.is_unique and (len(dev) == 1565) and (len(eva) == 1520)
    pred = (
        dict(np.load(ROOT / "private_data/refit_all_predictions.npz"))
        if args.inference_only
        else regenerate(dev, eva)
    )
    validate_original_bounds(pred, dev, eva)
    inference(pred, dev, eva)


if __name__ == "__main__":
    main()
