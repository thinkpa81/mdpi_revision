import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original_model_engine as eng

SEED = 2026102109
CAPITAL_CODES = (1, 4, 9)
REGION_NAMES = {
    1: "Seoul",
    2: "Busan",
    3: "Daegu",
    4: "Incheon",
    5: "Gwangju",
    6: "Daejeon",
    7: "Ulsan",
    8: "Sejong",
    9: "Gyeonggi",
    10: "Gangwon",
    11: "Chungbuk",
    12: "Chungnam",
    13: "Jeonbuk",
    14: "Jeonnam",
    15: "Gyeongbuk",
    16: "Gyeongnam",
    17: "Jeju",
}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def prepare(rawdir, package, private):
    source = (
        package
        / "original_results/main_result/mdpi-main/01_required_prior_analysis/code/harmonize_data.py"
    )
    spec = importlib.util.spec_from_file_location("region_harmonizer", source)
    harm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm)
    waves, source_records = ({}, [])
    for year in (2023, 2024, 2025):
        path = rawdir / f"{year}_data.xlsx"
        raw = harm.read_source(path)
        waves[year], quality = harm.harmonize(raw, year)
        source_records.append(
            {
                "year": year,
                "source_filename": path.name,
                "sha256": digest(path),
                "rows": len(raw),
                "columns": len(raw.columns),
            }
        )
        print(f"Read source wave {year}: {len(raw)} records", flush=True)
    dev = harm.transition(waves[2023], waves[2024], 2023)
    eva = harm.transition(waves[2024], waves[2025], 2024)
    full = []
    for by, oy in ((2023, 2024), (2024, 2025)):
        baseline = waves[by].copy()
        baseline["hesitation_frequency"] = 1
        full.append(harm.transition(baseline, waves[oy], by))
    for name, frame in (("development", dev), ("evaluation", eva)):
        frame.to_csv(private / f"{name}.csv", index=False)
    overlap_rows = []
    for population, d, e in (("restricted", dev, eva), ("full_linked_panel", *full)):
        shared = e.id.isin(set(d.id))
        overlap_rows.append(
            {
                "population": population,
                "development_n": len(d),
                "evaluation_n": len(e),
                "shared_n": int(shared.sum()),
                "disjoint_n": int((~shared).sum()),
                "shared_fraction": float(shared.mean()),
            }
        )
    return (dev, eva, source_records, overlap_rows, digest(source))


def bounds(contrasts, bootstrap_weights):
    point = contrasts.mean(axis=0)
    draws = bootstrap_weights @ contrasts
    centered = draws - point
    q_abs = np.quantile(np.max(np.abs(centered), axis=1), 0.95)
    q_low = np.quantile(np.max(centered, axis=1), 0.95)
    q_up = np.quantile(np.max(-centered, axis=1), 0.95)
    result = []
    for j, value in enumerate(point):
        result.append(
            {
                "delta_log_loss": float(value),
                "point_lower95_two_sided": float(value - np.quantile(centered[:, j], 0.975)),
                "point_upper95_two_sided": float(value - np.quantile(centered[:, j], 0.025)),
                "simultaneous_lower95_two_sided": float(value - q_abs),
                "simultaneous_upper95_two_sided": float(value + q_abs),
                "simultaneous_lower95_one_sided": float(value - q_low),
                "simultaneous_upper95_one_sided": float(value + q_up),
                "harm_detected_two_sided": bool(value - q_abs > 0),
                "benefit_detected_two_sided": bool(value + q_abs < 0),
            }
        )
    return result


def main():
    if not __debug__:
        raise RuntimeError("Run without -O or -OO so validation assertions remain active.")
    ap = argparse.ArgumentParser(
        description="Geographic coarsening and participant-overlap diagnostics from authorized raw waves.\n\nThe geographic rule maps Seoul/Incheon/Gyeonggi (raw codes 1/4/9) to capital area\nand all other valid regions to noncapital. It is a retrospective coding\nsensitivity, defined without evaluation-outcome optimization. D continues to\ncost four questionnaire items. Fixed D and full-information portfolios are\nretuned by development-only CV; this is not budget-portfolio reselection.\n\nOnly aggregate CSV/JSON files are written to --out. Row-level records and\npredictions are written only to the explicitly separate --private path.\n"
    )
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    assert not args.private.resolve().is_relative_to(args.out.resolve())
    start = time.time()
    dev, eva, sources, overlap_rows, harmonizer_hash = prepare(
        args.raw, args.package, args.private
    )
    assert len(dev) == 1565 and len(eva) == 1520
    shared = eva.id.isin(set(dev.id)).to_numpy()
    assert shared.sum() == 641
    masks = {"whole": np.ones(len(eva), bool), "shared": shared, "disjoint": ~shared}
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=eng.SEED).split(dev, dev.outcome_any_use)
    )
    frozen = json.loads((args.package / "revision/audit/original_frozen_configs.json").read_text())
    expected = {(r["algorithm"], r["subset"]): r for r in frozen}
    original_check, selections, predictions = ([], [], {})
    npred = eng.fit_predict(dev, eva, [], "ridge", 1)[0]
    benchmarks = {False: npred, True: eng.p0(dev, eva)}
    with threadpool_limits(limits=1):
        for coding in ("region17", "capital_noncapital"):
            d, e = (dev.copy(), eva.copy())
            if coding == "capital_noncapital":
                for frame in (d, e):
                    frame["region"] = (
                        frame.region.isin(CAPITAL_CODES).astype(float).where(frame.region.notna())
                    )
            for algorithm in ("ridge", "hgb"):
                for history in (False, True):
                    for portfolio in ("D", "full"):
                        blocks = (["H"] if history else []) + (
                            ["D"] if portfolio == "D" else eng.OPTIONAL
                        )
                        subset = eng.name(blocks)
                        spec = {
                            "blocks": blocks,
                            "subset": subset,
                            "has_history": history,
                            "additional_items": 4 if portfolio == "D" else 30,
                        }
                        selected, oof, _ = eng.cv_task(d, folds, spec, algorithm)
                        p, columns = eng.fit_predict(
                            d, e, blocks, algorithm, selected["best_config"]
                        )
                        key = (coding, algorithm, history, portfolio)
                        predictions[key] = p
                        if coding == "region17":
                            old = expected[algorithm, subset]
                            check = {
                                "algorithm": algorithm,
                                "history_available": history,
                                "portfolio": portfolio,
                                "selected_config_matches": selected["best_config"]
                                == old["best_config"],
                                "oof_loss_difference": selected["oof_log_loss"]
                                - old["oof_log_loss"],
                            }
                            original_check.append(check)
                            assert (
                                check["selected_config_matches"]
                                and abs(check["oof_loss_difference"]) < 1e-10
                            )
                        selections.append(
                            {
                                "region_coding": coding,
                                "algorithm": algorithm,
                                "history_available": history,
                                "portfolio": portfolio,
                                "subset": subset,
                                "additional_items": spec["additional_items"],
                                "encoded_columns": columns,
                                "selected_config": json.dumps(
                                    selected["best_config"], sort_keys=True
                                ),
                                "development_oof_log_loss": selected["oof_log_loss"],
                            }
                        )
                        print(f"Fitted {key}: {selected['best_config']}", flush=True)
    np.savez_compressed(
        args.private / "region_predictions.npz",
        **{"|".join(map(str, k)): p for k, p in predictions.items()},
    )
    eva[["id"]].to_csv(args.private / "prediction_ids.csv", index=False)
    pd.DataFrame(selections).to_csv(args.out / "region_development_selection.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(args.out / "overlap_sample_proportions.csv", index=False)
    counts = []
    for sn, f in (
        ("development", dev),
        ("evaluation", eva),
        ("evaluation_shared", eva.loc[shared]),
        ("evaluation_disjoint", eva.loc[~shared]),
    ):
        for code, label in REGION_NAMES.items():
            count = int(f.region.eq(code).sum())
            counts.append(
                {
                    "sample": sn,
                    "coding": "region17",
                    "code": code,
                    "label": label,
                    "n": count,
                    "denominator": len(f),
                    "proportion": count / len(f),
                }
            )
        for code, label in ((1, "Capital area"), (0, "Noncapital area")):
            count = int(((f.region.isin(CAPITAL_CODES) == bool(code)) & f.region.notna()).sum())
            counts.append(
                {
                    "sample": sn,
                    "coding": "capital_noncapital",
                    "code": code,
                    "label": label,
                    "n": count,
                    "denominator": len(f),
                    "proportion": count / len(f),
                }
            )
    pd.DataFrame(counts).to_csv(args.out / "region_category_distributions.csv", index=False)
    weights = (
        np.random.default_rng(SEED)
        .multinomial(len(eva), np.ones(len(eva)) / len(eva), size=args.bootstrap)
        .astype(float)
    )
    metrics, paired, overlap = ([], [], [])
    yall = eva.outcome_any_use.to_numpy(int)
    keys = list(predictions)
    for domain, mask in masks.items():
        y = yall[mask]
        w = weights[:, mask]
        w /= w.sum(axis=1, keepdims=True)
        losses = {k: eng.loss_vector(y, predictions[k][mask]) for k in keys}
        base_losses = {h: eng.loss_vector(y, p[mask]) for h, p in benchmarks.items()}
        family = np.column_stack([losses[k] - base_losses[k[2]] for k in keys])
        inference = bounds(family, w)
        for k, inf in zip(keys, inference):
            coding, algorithm, history, portfolio = k
            metrics.append(
                {
                    "domain": domain,
                    "n": len(y),
                    "events": int(y.sum()),
                    "region_coding": coding,
                    "algorithm": algorithm,
                    "history_available": history,
                    "portfolio": portfolio,
                    "log_loss": float(losses[k].mean()),
                    "benchmark_log_loss": float(base_losses[history].mean()),
                    "roc_auc": float(roc_auc_score(y, predictions[k][mask])),
                    "benchmark": "P0" if history else "NONE",
                    **inf,
                    "family_contrasts": 16,
                    "family": "16 original/coarsened model-versus-benchmark contrasts within evaluation domain",
                }
            )
        pairs = [(a, h, p) for a in ("ridge", "hgb") for h in (False, True) for p in ("D", "full")]
        diffs = np.column_stack(
            [losses["capital_noncapital", *k] - losses["region17", *k] for k in pairs]
        )
        for key, inf in zip(pairs, bounds(diffs, w)):
            algorithm, history, portfolio = key
            paired.append(
                {
                    "domain": domain,
                    "n": len(y),
                    "algorithm": algorithm,
                    "history_available": history,
                    "portfolio": portfolio,
                    **inf,
                    "direction": "coarsened minus 17-category log loss; negative favors coarsening",
                    "family_contrasts": 8,
                    "family": "8 paired coarsening contrasts within evaluation domain",
                }
            )
        fullkeys = [("region17", a, h, "full") for a in ("ridge", "hgb") for h in (False, True)]
        fcontrast = np.column_stack([losses[k] - base_losses[k[2]] for k in fullkeys])
        for key, inf in zip(fullkeys, bounds(fcontrast, w)):
            _, algorithm, history, _ = key
            overlap.append(
                {
                    "domain": domain,
                    "n": len(y),
                    "algorithm": algorithm,
                    "history_available": history,
                    "full_log_loss": float(losses[key].mean()),
                    "benchmark_log_loss": float(base_losses[history].mean()),
                    **inf,
                    "family_contrasts": 4,
                    "family": "4 original full-model-versus-benchmark contrasts within evaluation domain",
                }
            )
    pd.DataFrame(metrics).to_csv(args.out / "region_model_comparisons.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out / "region_coarsening_paired_contrasts.csv", index=False)
    pd.DataFrame(overlap).to_csv(args.out / "overlap_original_full_comparisons.csv", index=False)
    comparison = pd.DataFrame(paired).rename(
        columns={
            "delta_log_loss": "delta_coarsened_minus_original",
            "simultaneous_lower95_two_sided": "change_simultaneous_lower95",
            "simultaneous_upper95_two_sided": "change_simultaneous_upper95",
            "benefit_detected_two_sided": "coarsening_improvement_supported",
            "harm_detected_two_sided": "coarsening_deterioration_supported",
        }
    )
    join_keys = ["domain", "algorithm", "history_available", "portfolio"]
    metric_frame = pd.DataFrame(metrics)
    for coding, short in (("region17", "original"), ("capital_noncapital", "coarsened")):
        subset = metric_frame.loc[
            metric_frame.region_coding.eq(coding),
            join_keys
            + [
                "delta_log_loss",
                "log_loss",
                "simultaneous_lower95_two_sided",
                "simultaneous_upper95_two_sided",
            ],
        ]
        subset = subset.rename(
            columns={c: short + "_" + c for c in subset.columns if c not in join_keys}
        )
        comparison = comparison.merge(subset, on=join_keys, validate="one_to_one")
    keep = join_keys + [
        "n",
        "original_delta_log_loss",
        "coarsened_delta_log_loss",
        "delta_coarsened_minus_original",
        "change_simultaneous_lower95",
        "change_simultaneous_upper95",
        "coarsening_improvement_supported",
        "coarsening_deterioration_supported",
        "original_log_loss",
        "coarsened_log_loss",
        "original_simultaneous_lower95_two_sided",
        "original_simultaneous_upper95_two_sided",
        "coarsened_simultaneous_lower95_two_sided",
        "coarsened_simultaneous_upper95_two_sided",
        "family",
        "family_contrasts",
    ]
    comparison[keep].to_csv(args.out / "region_compact_comparison.csv", index=False)
    previous = pd.read_csv(args.package / "revision/tables/full_models_vs_benchmarks.csv")
    now = pd.DataFrame(overlap)
    chk = now.merge(
        previous, on=["domain", "algorithm", "history_available"], suffixes=("_new", "_old")
    )
    maxdiff = float((chk.delta_log_loss - chk.delta_full_vs_benchmark).abs().max())
    assert len(chk) == 8 and maxdiff < 1e-10
    report = {
        "status": "passed",
        "analysis": "retrospective region coding and participant-overlap sensitivity",
        "sources": sources,
        "harmonizer_sha256": harmonizer_hash,
        "script_sha256": digest(__file__),
        "model_engine_sha256": digest(eng.__file__),
        "sample_overlap": overlap_rows,
        "capital_region_codes": list(CAPITAL_CODES),
        "capital_region_labels": [REGION_NAMES[c] for c in CAPITAL_CODES],
        "geographic_rule": "Seoul, Incheon, Gyeonggi versus all other valid regions; missing values remain missing",
        "original_category_meanings_source": "revision/tables/demographic_raw_embedded_codebook_mapping.csv",
        "development_selection": "same original 5-fold stratified CV, seed 20260907 and original learner grids; evaluated outcomes never tune models",
        "portfolio_scope": "fixed D and full-information portfolios only; no budget-portfolio reselection",
        "models_fit": len(predictions),
        "original_models_reproduced": original_check,
        "compact_comparison_rows": len(comparison),
        "published_full_model_estimates_compared": len(chk),
        "maximum_absolute_published_estimate_difference": maxdiff,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": SEED,
        "bootstrap_scope": "paired evaluation participant bootstrap conditional on fitted predictions; shared/disjoint masks use the same whole-evaluation draws and normalize within subgroup",
        "intervals": "two-sided pointwise basic intervals and centered maximum-deviation simultaneous intervals; primary family sizes 16 model contrasts and 8 coarsening contrasts, separately in each evaluation domain; original full-model diagnostic has family size 4",
        "cautions": [
            "Coarsening was motivated by observed distribution changes and is retrospective.",
            "D still uses four raw questionnaire items; this changes representation, not questionnaire acquisition cost.",
            "Overlap subgroup differences do not identify the causal effect of participant reuse, restriction, or wording.",
            "Same evaluation-year wording across subgroups shows that wording change alone is not a sufficient empirical description of the observed heterogeneity, but interaction with composition remains possible.",
            "This finite sensitivity does not establish performance for other coarsening rules or new populations.",
            "The analysis repeats tuning for each coding but the inference conditions on the resulting fitted models; it does not include full training/reselection uncertainty.",
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "seconds": time.time() - start,
    }
    write_json(args.out / "region_overlap_analysis.json", report)
    print(
        json.dumps(
            {"status": "passed", "models_fit": len(predictions), "seconds": report["seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
