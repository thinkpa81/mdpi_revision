from __future__ import annotations
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import sys, json, time, argparse, hashlib, platform, warnings

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / "analysis/round2/model_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy, sklearn
import xgboost, lightgbm, catboost
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import run_model_sensitivity as ms
import task_provenance as provenance

SEED = 20261020
B = 5000
BOOT_SEED = 2026102031
P = ROOT / "private_data"
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
PARTS = P / "round2_boosting_parts"
PREDS = P / "round2_boosting_predictions"
REPS = ms.REPS
GRIDS = {
    "xgboost": [
        dict(n_estimators=n, max_depth=d, reg_lambda=l)
        for n in [100, 300]
        for d in [2, 4]
        for l in [1.0, 10.0, 100.0]
    ],
    "lightgbm": [
        dict(n_estimators=n, num_leaves=d, reg_lambda=l)
        for n in [100, 300]
        for d in [7, 15]
        for l in [1.0, 10.0, 100.0]
    ],
    "catboost": [
        dict(iterations=n, depth=d, l2_leaf_reg=l)
        for n in [100, 300]
        for d in [3, 5]
        for l in [3.0, 30.0, 100.0]
    ],
}
FIXED = {
    "xgboost": dict(
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        min_child_weight=10.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        max_bin=256,
        n_jobs=1,
        random_state=SEED,
        verbosity=0,
    ),
    "lightgbm": dict(
        learning_rate=0.05,
        objective="binary",
        boosting_type="gbdt",
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        max_bin=255,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        random_state=SEED,
        verbosity=-1,
    ),
    "catboost": dict(
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        task_type="CPU",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_strength=1.0,
        border_count=254,
        one_hot_max_size=2,
        thread_count=1,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    ),
}


def load(pop):
    dev = pd.read_csv(
        P / ("development.csv" if pop == "restricted" else "full_development.csv"),
        dtype={"id": str},
    )
    ev = pd.read_csv(
        P / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"), dtype={"id": str}
    )
    assert dev.id.is_unique and ev.id.is_unique
    return (dev, ev)


def groups(dev, ev):
    result = {"all": np.ones(len(ev), bool), "person_disjoint": ~ev.id.isin(dev.id).to_numpy()}
    if "hesitation_frequency" in ev:
        result.update(
            hesitant=ev.hesitation_frequency.isin([1, 2]).to_numpy(),
            other=ev.hesitation_frequency.isin([3, 4, 9]).to_numpy(),
            low_or_no_hesitation=ev.hesitation_frequency.isin([3, 4]).to_numpy(),
            no_service_experience=ev.hesitation_frequency.eq(9).to_numpy(),
        )
        result = {g: m for g, m in result.items() if m.sum() > 1}
    return result


def matrix_pair(train, test, blocks, algo):
    if algo != "catboost":
        prep = ms.orig.make_model(blocks, "ridge", 1).named_steps["prep"]
        return (prep.fit_transform(train), prep.transform(test), [])
    num = sum([ms.orig.BLOCKS[b]["numeric"] for b in blocks], [])
    cat = sum([ms.orig.BLOCKS[b]["categorical"] for b in blocks], [])
    a = pd.DataFrame(index=train.index)
    b = pd.DataFrame(index=test.index)
    for c in num:
        median = float(train[c].median())
        assert np.isfinite(median)
        a[c] = train[c].fillna(median).astype(float)
        b[c] = test[c].fillna(median).astype(float)
    for c in cat:
        mode = train[c].dropna().mode().iloc[0]
        a[c] = train[c].fillna(mode).astype(str)
        b[c] = test[c].fillna(mode).astype(str)
    return (a, b, cat)


def model(algo, cfg, cat):
    kw = dict(FIXED[algo], **cfg)
    if algo == "xgboost":
        return xgboost.XGBClassifier(**kw)
    if algo == "lightgbm":
        return lightgbm.LGBMClassifier(**kw)
    return catboost.CatBoostClassifier(**kw, cat_features=cat)


def task_cache(pop, algo, rep):
    key = f"{pop}__{algo}__{rep}"
    inputs = {
        "development": P / ("development.csv" if pop == "restricted" else "full_development.csv"),
        "evaluation": P / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
    }
    outputs = {
        "predictions_and_oof": PREDS / (key + ".npz"),
        "metrics": PARTS / (key + "_metrics.csv"),
        "tuning": PARTS / (key + "_tuning.csv"),
        "fitted_parameters": PARTS / (key + "_params.json"),
    }
    cats = (
        sum([ms.orig.BLOCKS[b]["categorical"] for b in REPS[rep]], [])
        if algo == "catboost"
        else []
    )
    spec = provenance.build_spec(
        key,
        inputs,
        {
            "round2_boosting": Path(__file__),
            "run_model_sensitivity": Path(ms.__file__),
            "run_exhaustive": Path(ms.orig.__file__),
        },
        dict(
            population=pop,
            algorithm=algo,
            representation=rep,
            blocks=REPS[rep],
            grid=GRIDS[algo],
            fixed_parameters=FIXED[algo],
            categorical_features=cats,
            resolved_estimator_parameters=[
                model(algo, c, cats).get_params(deep=False) for c in GRIDS[algo]
            ],
            seed=SEED,
            folds=5,
            shuffle=True,
            criterion="pooled OOF log loss",
            tie_rule="first grid member",
        ),
        dict(
            ms.cache_libraries(),
            xgboost=xgboost.__version__,
            lightgbm=lightgbm.__version__,
            catboost=catboost.__version__,
        ),
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


def task(pop, algo, rep):
    receipt, task_spec, task_outputs = task_cache(pop, algo, rep)
    if provenance.reusable(receipt, task_spec, task_outputs):
        return
    start = time.time()
    dev, ev = load(pop)
    blocks = REPS[rep]
    y = dev.outcome_any_use.to_numpy(int)
    ye = ev.outcome_any_use.to_numpy(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, y))
    matrices = []
    for tr, va in folds:
        assert set(dev.id.iloc[tr]).isdisjoint(dev.id.iloc[va])
        matrices.append((tr, va, *matrix_pair(dev.iloc[tr], dev.iloc[va], blocks, algo)))
    rows = []
    oofs = []
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for i, cfg in enumerate(GRIDS[algo]):
            pcv = np.zeros(len(dev))
            fold_loss = []
            for tr, va, xtr, xva, cat in matrices:
                fit = model(algo, cfg, cat).fit(xtr, y[tr])
                pcv[va] = fit.predict_proba(xva)[:, 1]
                fold_loss.append(float(ms.orig.loss_vector(y[va], pcv[va]).mean()))
            rows.append(
                dict(
                    population=pop,
                    algorithm=algo,
                    representation=rep,
                    configuration_index=i,
                    params=json.dumps(cfg, sort_keys=True),
                    development_cv_log_loss=float(ms.orig.loss_vector(y, pcv).mean()),
                    fold_log_losses=json.dumps(fold_loss),
                )
            )
            oofs.append(pcv)
        choice = int(np.argmin([r["development_cv_log_loss"] for r in rows]))
        cfg = GRIDS[algo][choice]
        xtr, xte, cat = matrix_pair(dev, ev, blocks, algo)
        fit = model(algo, cfg, cat).fit(xtr, y)
        pred = fit.predict_proba(xte)[:, 1]
    assert np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
    key = f"{pop}__{algo}__{rep}"
    np.savez_compressed(
        PREDS / (key + ".npz"),
        prediction=pred,
        development_oof=oofs[choice],
        evaluation_id=ev.id.to_numpy(),
        development_id=dev.id.to_numpy(),
    )
    for i, r in enumerate(rows):
        r["selected"] = i == choice
    duration = time.time() - start
    metrics = []
    for group, sel in groups(dev, ev).items():
        metrics.append(
            dict(
                population=pop,
                algorithm=algo,
                representation=rep,
                evaluation_group=group,
                selected_configuration=choice,
                selected_params=json.dumps(cfg, sort_keys=True),
                development_cv_log_loss=rows[choice]["development_cv_log_loss"],
                elapsed_seconds=duration,
                **ms.met(ye[sel], pred[sel]),
            )
        )
    pd.DataFrame(rows).to_csv(PARTS / (key + "_tuning.csv"), index=False)
    pd.DataFrame(metrics).to_csv(PARTS / (key + "_metrics.csv"), index=False)
    (PARTS / (key + "_params.json")).write_text(
        json.dumps(fit.get_params(), indent=2, default=str)
    )
    if task_cache(pop, algo, rep)[1] != task_spec:
        raise provenance.CacheProvenanceError(
            f"Inputs or computation changed during task {key}; no receipt committed"
        )
    provenance.write_receipt(receipt, task_spec, task_outputs)
    print(key, "complete", round(duration, 2), "seconds", flush=True)


def aggregate():
    verify_task_caches(require_complete=True)
    metrics = []
    tuning = []
    for pop in ["restricted", "full"]:
        for algo in GRIDS:
            for rep in REPS:
                key = f"{pop}__{algo}__{rep}"
                metrics.append(pd.read_csv(PARTS / (key + "_metrics.csv")))
                tuning.append(pd.read_csv(PARTS / (key + "_tuning.csv")))
    pd.concat(metrics, ignore_index=True).to_csv(T / "round2_boosting_metrics.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(T / "round2_boosting_tuning.csv", index=False)


def intervals():
    verify_task_caches(require_complete=True)
    output = []
    families = []
    for ip, pop in enumerate(["restricted", "full"]):
        dev, ev = load(pop)
        y = ev.outcome_any_use.to_numpy(int)
        refs = {
            "P0": ms.orig.p0(dev, ev),
            "NONE": np.full(len(ev), (dev.outcome_any_use.sum() + 1) / (len(dev) + 2)),
        }
        llref = {k: ms.orig.loss_vector(y, p) for k, p in refs.items()}
        columns = []
        diffs = []
        designated = []
        for algo in GRIDS:
            for rep in REPS:
                data = np.load(PREDS / f"{pop}__{algo}__{rep}.npz", allow_pickle=True)
                assert np.array_equal(data["evaluation_id"].astype(str), ev.id.to_numpy())
                ll = ms.orig.loss_vector(y, data["prediction"])
                for ref in refs:
                    j = len(columns)
                    columns.append(
                        dict(population=pop, algorithm=algo, representation=rep, benchmark=ref)
                    )
                    diffs.append(ll - llref[ref])
                    if (rep, ref) in [("full_H", "P0"), ("full_noH", "NONE")]:
                        designated.append(j)
        assert len(designated) == 6
        D = np.asarray(diffs).T
        for ig, (group, mask) in enumerate(groups(dev, ev).items()):
            d = D[mask]
            n = len(d)
            mean = d.mean(axis=0)
            rng = np.random.default_rng(BOOT_SEED + 100 * ip + ig)
            boot = []
            with threadpool_limits(1):
                for j in range(0, B, 200):
                    w = rng.multinomial(n, np.full(n, 1 / n), size=min(200, B - j))
                    boot.append(w @ d / n)
            boot = np.concatenate(boot)
            lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
            critical = float(np.quantile(np.abs((boot - mean)[:, designated]).max(axis=1), 0.95))
            families.append(
                dict(
                    population=pop,
                    evaluation_group=group,
                    n=n,
                    replicates=B,
                    family_size=6,
                    critical=critical,
                    family="3 added algorithms x [full_H vs P0, full_noH vs NONE]",
                    statistic="maximum absolute centered paired bootstrap mean loss difference",
                )
            )
            for j, meta in enumerate(columns):
                output.append(
                    dict(
                        **meta,
                        evaluation_group=group,
                        n=n,
                        replicates=B,
                        delta_log_loss=mean[j],
                        pointwise_low=lo[j],
                        pointwise_high=hi[j],
                        bootstrap_se=boot[:, j].std(ddof=1),
                        in_full_model_family=j in designated,
                        simultaneous_low=mean[j] - critical if j in designated else np.nan,
                        simultaneous_high=mean[j] + critical if j in designated else np.nan,
                    )
                )
    df = pd.DataFrame(output)
    df.to_csv(T / "round2_boosting_paired_bootstrap.csv", index=False)
    df[df.in_full_model_family].to_csv(T / "round2_boosting_full_model_intervals.csv", index=False)
    pd.DataFrame(families).to_csv(T / "round2_boosting_inference_families.csv", index=False)


def manifest(status, start):
    info = dict(
        status=status,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        elapsed_seconds=time.time() - start,
        scope="post-review finite comparison; eight fixed information representations, no budget or subset search",
        evaluation_used_for_tuning=False,
        folds=5,
        seed=SEED,
        representations=REPS,
        grids=GRIDS,
        fixed_parameters=FIXED,
        expected_final_models=48,
        expected_cv_fits=2880,
        expected_total_fits=2928,
        tuning="pooled development-only five-fold out-of-fold log loss; first grid member wins exact ties",
        preprocessing="XGBoost/LightGBM fold-fit median+standardization numeric and modal+onehot categorical; CatBoost fold-fit median numeric and modal categorical strings, native categorical handling and Ordered boosting",
        early_stopping=False,
        class_weights=False,
        oversampling=False,
        inference=dict(
            replicates=B,
            seed=BOOT_SEED,
            conditional_fixed_predictions=True,
            selection_repeated=False,
            pointwise="95% percentile paired bootstrap",
            simultaneous="observed difference plus or minus 95th percentile maximum absolute centered bootstrap deviation",
            family="six new-algorithm full-model contrasts separately within each population and evaluation group; original four-algorithm eight-contrast families unchanged",
        ),
        libraries={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        input_sha256={
            f: hashlib.sha256((P / f).read_bytes()).hexdigest()
            for f in [
                "development.csv",
                "evaluation.csv",
                "full_development.csv",
                "full_evaluation.csv",
            ]
        },
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (A / "round2_boosting_manifest.json").write_text(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--mode", choices=["pilot", "fit", "aggregate", "intervals", "all"], default="all"
    )
    args = parser.parse_args()
    for path in [T, A, PARTS, PREDS]:
        path.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.mode == "pilot":
        jobs = verify_task_caches(tasks=[("restricted", algo, "full_H") for algo in GRIDS])
        for task_args in jobs:
            task(*task_args)
        return
    jobs = verify_task_caches(require_complete=args.mode not in ["fit", "all"])
    manifest("started", start)
    if args.mode in ["fit", "all"]:
        print("remaining tasks", len(jobs), flush=True)
        Parallel(n_jobs=args.jobs)((delayed(task)(*t) for t in jobs))
    if args.mode in ["fit", "aggregate", "all"]:
        aggregate()
    if args.mode in ["intervals", "all"]:
        intervals()
    manifest("completed_" + args.mode, start)


if __name__ == "__main__":
    main()
