from __future__ import annotations
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import argparse, json, time, sys, hashlib, platform
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from importlib.metadata import version as package_version
import task_provenance as provenance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(ROOT / "original_results/main_result/mdpi-main/01_required_prior_analysis/code")
)
import run_exhaustive as orig

SEED = 20261020
TABLES = ROOT / "revision/tables"
PRIVATE = ROOT / "private_data"
AUDIT = ROOT / "revision/audit"
PARTS = PRIVATE / "model_sensitivity_parts"
REPS = {
    "H": ["H"],
    "B": ["B"],
    "H_B": ["H", "B"],
    "D": ["D"],
    "H_D": ["H", "D"],
    "H_D_B": ["H", "D", "B"],
    "full_H": ["H", "D", "B", "M", "P", "C", "A", "K"],
    "full_noH": ["D", "B", "M", "P", "C", "A", "K"],
}
GRIDS = {
    "ridge": [{"C": float(c)} for c in np.logspace(-4, 2, 13)],
    "hgb": [
        {"max_iter": it, "max_leaf_nodes": leaf, "learning_rate": rate}
        for it in [100, 300]
        for leaf in [3, 7, 15, 31]
        for rate in [0.03, 0.05, 0.1]
    ],
    "rf": [
        {"min_samples_leaf": leaf, "max_features": feat}
        for leaf in [5, 20, 50]
        for feat in [1.0, "sqrt"]
    ],
    "extratrees": [
        {"min_samples_leaf": leaf, "max_features": feat}
        for leaf in [5, 20, 50]
        for feat in [1.0, "sqrt"]
    ],
}


def model(algo, cfg):
    if algo == "ridge":
        return LogisticRegression(**cfg, max_iter=3000, solver="lbfgs", random_state=SEED)
    if algo == "hgb":
        return HistGradientBoostingClassifier(
            **cfg,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED,
        )
    cls = RandomForestClassifier if algo == "rf" else ExtraTreesClassifier
    return cls(**cfg, n_estimators=200, max_depth=None, n_jobs=1, random_state=SEED)


def cache_libraries():
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=orig.scipy.__version__,
        sklearn=orig.sklearn.__version__,
        joblib=orig.joblib.__version__,
        threadpoolctl=package_version("threadpoolctl"),
    )


def task_cache(pop, algo, rep):
    key = f"{pop}__{algo}__{rep}"
    inputs = {
        "development": PRIVATE
        / ("development.csv" if pop == "restricted" else "full_development.csv"),
        "evaluation": PRIVATE
        / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
    }
    outputs = {
        "predictions_and_oof": PRIVATE / (key + ".npz"),
        "metrics": PARTS / (key + "_metrics.csv"),
        "tuning": PARTS / (key + "_grid.csv"),
    }
    spec = provenance.build_spec(
        key,
        inputs,
        {"run_model_sensitivity": Path(__file__), "run_exhaustive": Path(orig.__file__)},
        dict(
            population=pop,
            algorithm=algo,
            representation=rep,
            blocks=REPS[rep],
            grid=GRIDS[algo],
            resolved_estimator_parameters=[
                model(algo, c).get_params(deep=False) for c in GRIDS[algo]
            ],
            seed=SEED,
            folds=5,
            shuffle=True,
            criterion="pooled OOF log loss",
            tie_rule="first grid member",
        ),
        cache_libraries(),
    )
    return (PARTS / (key + "_provenance.json"), spec, outputs)


def verify_task_caches(require_complete=False, tasks=None):
    pending = []
    if tasks is None:
        tasks = [(p, a, r) for p in ["restricted", "full"] for a in GRIDS for r in REPS]
    for task_args in tasks:
        receipt, spec, outputs = task_cache(*task_args)
        if not provenance.reusable(receipt, spec, outputs):
            if require_complete:
                raise provenance.CacheProvenanceError(f"Missing completed task: {receipt}")
            pending.append(task_args)
    return pending


def auc_ci(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pos = p[y == 1]
    neg = p[y == 0]
    m = len(pos)
    n = len(neg)
    if m < 2 or n < 2:
        return (np.nan, np.nan, np.nan)
    r = rankdata(np.r_[pos, neg])
    rp = rankdata(pos)
    rn = rankdata(neg)
    v1 = (r[:m] - rp) / n
    v0 = 1 - (r[m:] - rn) / m
    auc = v1.mean()
    se = np.sqrt(v1.var(ddof=1) / m + v0.var(ddof=1) / n)
    return (
        float(auc),
        float(max(0, auc - 1.95996398454 * se)),
        float(min(1, auc + 1.95996398454 * se)),
    )


def met(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    loss = orig.loss_vector(y, p)
    auc, lo, hi = auc_ci(y, p)
    return dict(
        n=len(y),
        events=int(y.sum()),
        prevalence=float(y.mean()),
        log_loss=float(loss.mean()),
        brier=float(brier_score_loss(y, p)),
        roc_auc=auc,
        auc_ci_low=lo,
        auc_ci_high=hi,
        auc_ci_method="DeLong normal 95% conditional on fitted model",
        average_precision=float(average_precision_score(y, p)),
    )


def tuned_task(pop, algo, rep):
    receipt, task_spec, task_outputs = task_cache(pop, algo, rep)
    if provenance.reusable(receipt, task_spec, task_outputs):
        return (
            pd.read_csv(task_outputs["metrics"]).to_dict("records"),
            pd.read_csv(task_outputs["tuning"]).to_dict("records"),
        )
    start = time.time()
    blocks = REPS[rep]
    dev = pd.read_csv(
        PRIVATE / ("development.csv" if pop == "restricted" else "full_development.csv"),
        dtype={"id": str},
    )
    ev = pd.read_csv(
        PRIVATE / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
        dtype={"id": str},
    )
    y = dev.outcome_any_use.to_numpy(dtype=int)
    ye = ev.outcome_any_use.to_numpy(dtype=int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, y))
    matrices = []
    for tr, va in folds:
        assert set(dev.id.iloc[tr]).isdisjoint(dev.id.iloc[va])
        prep = orig.make_model(blocks, "ridge", 1).named_steps["prep"]
        matrices.append((tr, va, prep.fit_transform(dev.iloc[tr]), prep.transform(dev.iloc[va])))
    scores = []
    oofs = []
    with threadpool_limits(1):
        for config in GRIDS[algo]:
            p = np.zeros(len(dev))
            for tr, va, xtr, xva in matrices:
                fit = model(algo, config).fit(xtr, y[tr])
                p[va] = fit.predict_proba(xva)[:, 1]
            scores.append(float(orig.loss_vector(y, p).mean()))
            oofs.append(p)
        choice = int(np.argmin(scores))
        cfg = GRIDS[algo][choice]
        prep = orig.make_model(blocks, "ridge", 1).named_steps["prep"]
        train = prep.fit_transform(dev)
        test = prep.transform(ev)
        fit = model(algo, cfg).fit(train, y)
        p = fit.predict_proba(test)[:, 1]
    key = f"{pop}__{algo}__{rep}"
    np.savez_compressed(
        PRIVATE / (key + ".npz"),
        prediction=p,
        development_oof=oofs[choice],
        evaluation_id=ev.id.to_numpy(),
        development_id=dev.id.to_numpy(),
    )
    rows = []
    groups = {"all": np.ones(len(ev), bool), "person_disjoint": ~ev.id.isin(dev.id).to_numpy()}
    if pop == "full":
        groups.update(
            hesitant=ev.hesitation_frequency.isin([1, 2]).to_numpy(),
            other=~ev.hesitation_frequency.isin([1, 2]).to_numpy(),
        )
    p0 = orig.p0(dev, ev)
    none = np.repeat((y.sum() + 1) / (len(y) + 2), len(ev))
    for group, sel in groups.items():
        row = dict(
            population=pop,
            algorithm=algo,
            representation=rep,
            evaluation_group=group,
            selected_params=json.dumps(cfg, sort_keys=True),
            development_cv_log_loss=scores[choice],
            **met(ye[sel], p[sel]),
        )
        for bench, bp in [("P0", p0), ("NONE", none)]:
            d = orig.loss_vector(ye[sel], p[sel]) - orig.loss_vector(ye[sel], bp[sel])
            se = d.std(ddof=1) / np.sqrt(len(d))
            row.update(
                {
                    f"delta_vs_{bench}": d.mean(),
                    f"delta_vs_{bench}_low": d.mean() - 1.96 * se,
                    f"delta_vs_{bench}_high": d.mean() + 1.96 * se,
                }
            )
        rows.append(row)
    grid = [
        dict(
            population=pop,
            algorithm=algo,
            representation=rep,
            params=json.dumps(c, sort_keys=True),
            development_cv_log_loss=s,
            selected=i == choice,
        )
        for i, (c, s) in enumerate(zip(GRIDS[algo], scores))
    ]
    pd.DataFrame(rows).to_csv(PARTS / (key + "_metrics.csv"), index=False)
    pd.DataFrame(grid).to_csv(PARTS / (key + "_grid.csv"), index=False)
    if task_cache(pop, algo, rep)[1] != task_spec:
        raise provenance.CacheProvenanceError(
            f"Inputs or computation changed during task {key}; no receipt committed"
        )
    provenance.write_receipt(receipt, task_spec, task_outputs)
    print(key, "complete", round(time.time() - start, 1), "sec", flush=True)
    return (rows, grid)


def benchmarks():
    out = []
    for pop in ["restricted", "full"]:
        dev = pd.read_csv(
            PRIVATE / ("development.csv" if pop == "restricted" else "full_development.csv"),
            dtype={"id": str},
        )
        ev = pd.read_csv(
            PRIVATE / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
            dtype={"id": str},
        )
        y = dev.outcome_any_use.to_numpy(dtype=int)
        ye = ev.outcome_any_use.to_numpy(dtype=int)
        groups = {"all": np.ones(len(ev), bool), "person_disjoint": ~ev.id.isin(dev.id).to_numpy()}
        if pop == "full":
            groups.update(
                hesitant=ev.hesitation_frequency.isin([1, 2]).to_numpy(),
                other=~ev.hesitation_frequency.isin([1, 2]).to_numpy(),
            )
        for name, p in [
            ("P0", orig.p0(dev, ev)),
            ("NONE", np.repeat((y.sum() + 1) / (len(y) + 2), len(ev))),
        ]:
            for g, sel in groups.items():
                out.append(
                    dict(
                        population=pop,
                        algorithm="benchmark",
                        representation=name,
                        evaluation_group=g,
                        **met(ye[sel], p[sel]),
                    )
                )
    pd.DataFrame(out).to_csv(TABLES / "model_sensitivity_benchmarks.csv", index=False)


def existing_hd():
    dev = pd.read_csv(PRIVATE / "development.csv", dtype={"id": str})
    ev = pd.read_csv(PRIVATE / "evaluation.csv", dtype={"id": str})
    y = ev.outcome_any_use.to_numpy(dtype=int)
    frozen = json.loads(
        (
            ROOT
            / "original_results/main_result/mdpi-main/01_required_prior_analysis/analysis_outputs/frozen_development_selection.json"
        ).read_text()
    )
    allrows = frozen["all_models"]
    policies = []
    out = []
    preds = {}
    p0 = orig.p0(dev, ev)
    for algo in ["ridge", "hgb"]:
        candidates = [
            dict(
                r,
                incremental_cost=sum(
                    (orig.BLOCKS[b]["raw_items"] for b in r["blocks"] if b not in ["H", "D"])
                ),
            )
            for r in allrows
            if r["algorithm"] == algo
        ]
        folds = list(
            StratifiedKFold(5, shuffle=True, random_state=orig.SEED).split(
                dev, dev.outcome_any_use
            )
        )
        poof = np.zeros(len(dev))
        for tr, va in folds:
            poof[va] = orig.p0(dev.iloc[tr], dev.iloc[va])
        candidates.append(
            dict(
                subset="P0",
                blocks=["H"],
                best_config=None,
                oof_log_loss=float(orig.loss_vector(dev.outcome_any_use, poof).mean()),
                incremental_cost=0,
            )
        )
        hd = next((r for r in candidates if r["subset"] == "H_D"))
        ph, _ = orig.fit_predict(dev, ev, hd["blocks"], algo, hd["best_config"])
        preds[algo + "::H_D"] = ph
        for scenario in ["available_optional_use", "fixed_H_D_inclusion"]:
            pool = (
                candidates
                if scenario == "available_optional_use"
                else [r for r in candidates if "H" in r["blocks"] and "D" in r["blocks"]]
            )
            choices = [
                min(
                    (r for r in pool if r["incremental_cost"] <= b),
                    key=lambda r: (r["oof_log_loss"], r["incremental_cost"], r["subset"]),
                )
                for b in range(27)
            ]
            zero = choices[0]
            for r in choices:
                key = algo + "::" + r["subset"]
                if key not in preds:
                    preds[key] = (
                        p0
                        if r["subset"] == "P0"
                        else orig.fit_predict(dev, ev, r["blocks"], algo, r["best_config"])[0]
                    )
            pzero = preds[algo + "::" + zero["subset"]]
            for b, r in enumerate(choices):
                p = preds[algo + "::" + r["subset"]]
                meta = dict(
                    scenario=scenario,
                    algorithm=algo,
                    budget=b,
                    selected_subset=r["subset"],
                    actual_additional_cost=r["incremental_cost"],
                    development_cv_log_loss=r["oof_log_loss"],
                    params=json.dumps(r["best_config"]),
                    zero_cost_selected_subset=zero["subset"],
                    fixed_HD_params=json.dumps(hd["best_config"]),
                )
                for group, sel in [
                    ("all", np.ones(len(ev), bool)),
                    ("person_disjoint", ~ev.id.isin(dev.id).to_numpy()),
                ]:
                    item = dict(**meta, evaluation_group=group, **met(y[sel], p[sel]))
                    for label, bp in [("zero_cost_selected", pzero), ("fixed_HD", ph), ("P0", p0)]:
                        d = orig.loss_vector(y[sel], p[sel]) - orig.loss_vector(y[sel], bp[sel])
                        se = d.std(ddof=1) / np.sqrt(len(d))
                        item.update(
                            {
                                f"delta_vs_{label}": d.mean(),
                                f"delta_vs_{label}_low": d.mean() - 1.96 * se,
                                f"delta_vs_{label}_high": d.mean() + 1.96 * se,
                                f"{label}_log_loss": float(
                                    orig.loss_vector(y[sel], bp[sel]).mean()
                                ),
                            }
                        )
                    out.append(item)
                policies.append(meta)
    np.savez_compressed(PRIVATE / "existing_HD_predictions.npz", **preds)
    pd.DataFrame(out).to_csv(TABLES / "existing_HD_budget_frontier.csv", index=False)
    pd.DataFrame(policies).to_csv(TABLES / "existing_HD_development_selection.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mode", choices=["all", "hd", "aggregate"], default="all")
    args = parser.parse_args()
    TABLES.mkdir(exist_ok=True, parents=True)
    AUDIT.mkdir(exist_ok=True, parents=True)
    PARTS.mkdir(exist_ok=True, parents=True)
    if args.mode == "hd":
        existing_hd()
        return
    tasks = verify_task_caches(require_complete=args.mode == "aggregate")
    if args.mode == "all":
        print("TASKS", len(tasks), flush=True)
        Parallel(n_jobs=args.jobs)((delayed(tuned_task)(*x) for x in tasks))
    verify_task_caches(require_complete=True)
    rows = []
    grids = []
    for p in ["restricted", "full"]:
        for a in GRIDS:
            for r in REPS:
                rows.append(pd.read_csv(PARTS / f"{p}__{a}__{r}_metrics.csv"))
                grids.append(pd.read_csv(PARTS / f"{p}__{a}__{r}_grid.csv"))
    pd.concat(rows, ignore_index=True).to_csv(
        TABLES / "model_sensitivity_metrics.csv", index=False
    )
    pd.concat(grids, ignore_index=True).to_csv(
        TABLES / "model_sensitivity_tuning.csv", index=False
    )
    benchmarks()
    existing_hd()
    import sklearn

    info = {
        "status": "completed",
        "analysis": "post-review sensitivity, existing evaluation data revisited",
        "evaluation_used_for_selection": False,
        "folds": 5,
        "seed": SEED,
        "grids": GRIDS,
        "representations": REPS,
        "algorithms": ["ridge", "hgb", "rf", "extratrees"],
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "auc_ci": "DeLong normal 95 percent, conditional on fitted model",
        "loss_difference_ci": "paired normal approximation 95 percent, unadjusted sensitivity diagnostics",
        "additional_cost_scenario": "H and D already available with optional-use and fixed-inclusion variants; original-grid exhaustive search; 0 to 26 items",
        "input_sha256": {
            f: hashlib.sha256((PRIVATE / f).read_bytes()).hexdigest()
            for f in [
                "development.csv",
                "evaluation.csv",
                "full_development.csv",
                "full_evaluation.csv",
            ]
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (AUDIT / "model_sensitivity_manifest.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
