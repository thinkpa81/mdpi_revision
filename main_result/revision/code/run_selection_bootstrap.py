from pathlib import Path
import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import sys, json, time, argparse, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import original_model_engine as eng
import bootstrap_cache as cache

BLOCK_ORDER = list(eng.BLOCKS)
BUDGETS = [1, 4]
SPECS = [s for s in eng.configurations() if s["additional_items"] <= 4]
assert len(SPECS) == 34
FULLS = [s for s in eng.configurations() if s["additional_items"] == 30]
assert len(FULLS) == 2
FIT_SPECS = SPECS + FULLS
GRIDS = {"ridge": eng.RIDGE_GRID, "hgb": eng.HGB_GRID}


def loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def prepare(train, target):
    prep = eng.make_model(BLOCK_ORDER, "ridge", 0.1).named_steps["prep"]
    xt = prep.fit_transform(train)
    xv = prep.transform(target)
    cols = {}
    num = sum([eng.BLOCKS[b]["numeric"] for b in BLOCK_ORDER], [])
    for b in BLOCK_ORDER:
        cols[b] = [num.index(x) for x in eng.BLOCKS[b]["numeric"]]
    pos = len(num)
    cats = sum([eng.BLOCKS[b]["categorical"] for b in BLOCK_ORDER], [])
    catsizes = [len(x) for x in prep.named_transformers_["cat"].named_steps["encode"].categories_]
    for col, n in zip(cats, catsizes):
        block = next((b for b in BLOCK_ORDER if col in eng.BLOCKS[b]["categorical"]))
        cols[block] += list(range(pos, pos + n))
        pos += n
    return (xt, xv, cols)


def select_columns(blocks, cols):
    return sorted(sum([cols[b] for b in blocks], []))


def estimator(algo, cfg, seed):
    if algo == "ridge":
        return LogisticRegression(C=float(cfg), solver="lbfgs", max_iter=2000, random_state=seed)
    return HistGradientBoostingClassifier(
        **cfg,
        max_iter=100,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def fit_array(algo, cfg, seed, trainx, y, testx):
    model = estimator(algo, cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(trainx, y)
    return model.predict_proba(testx)[:, 1]


def build_folds(base, draw, seed):
    unique = np.unique(draw)
    y = base.outcome_any_use.to_numpy(int)
    folds = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, y[unique]):
        train_unique = set(unique[tr])
        valid_unique = set(unique[va])
        train = np.array([i for i in draw if i in train_unique], int)
        valid = np.array([i for i in draw if i in valid_unique], int)
        assert set(base.id.iloc[train]).isdisjoint(set(base.id.iloc[valid]))
        folds.append((train, valid))
    assert sum((len(v) for _, v in folds)) == len(draw)
    return folds


def evaluate(rep, dev, eva, union, base_seed, outdir, point=False):
    start = time.time()
    seed = base_seed + rep
    rng = np.random.default_rng(seed)
    if point:
        dd = np.arange(len(dev))
        ee = np.arange(len(eva))
        fold_seed = eng.SEED
    else:
        counts = rng.multinomial(len(union), np.repeat(1 / len(union), len(union)))
        lookup = dict(zip(union, counts))
        dd = np.repeat(np.arange(len(dev)), [lookup[x] for x in dev.id])
        ee = np.repeat(np.arange(len(eva)), [lookup[x] for x in eva.id])
        fold_seed = seed
    assert len(dd) > 0 and len(ee) > 0
    folds = build_folds(dev, dd, fold_seed)
    fold_data = []
    for tr, va in folds:
        train = dev.iloc[tr]
        valid = dev.iloc[va]
        xt, xv, cols = prepare(train, valid)
        fold_data.append(
            (
                train,
                valid,
                xt,
                xv,
                cols,
                valid.outcome_any_use.to_numpy(int),
                train.outcome_any_use.to_numpy(int),
            )
        )
    train = dev.iloc[dd]
    target = eva.iloc[ee]
    y = target.outcome_any_use.to_numpy(int)
    xt, xv, cols = prepare(train, target)
    ty = train.outcome_any_use.to_numpy(int)
    baseline = {
        "NONE": np.repeat((ty.sum() + 1) / (len(ty) + 2), len(target)),
        "P0": eng.p0(train, target),
    }
    cvbase = {}
    for name in ["NONE", "P0"]:
        cvbase[name] = sum(
            (
                loss(
                    vy,
                    (
                        np.repeat((try_.sum() + 1) / (len(try_) + 2), len(vy))
                        if name == "NONE"
                        else eng.p0(tr, va)
                    ),
                ).sum()
                for tr, va, _, _, _, vy, try_ in fold_data
            )
        ) / len(dd)
    rows = []
    models = {}
    candidates = []
    for algo, grid in GRIDS.items():
        for spec in FIT_SPECS:
            subset = spec["subset"]
            blocks = spec["blocks"]
            if not blocks:
                bestcfg = None
                cvll = cvbase["NONE"]
                p = baseline["NONE"]
            else:
                scores = []
                for cfg in grid:
                    total = 0.0
                    for tr, va, xtr, xva, cmap, vy, try_ in fold_data:
                        cc = select_columns(blocks, cmap)
                        pp = fit_array(algo, cfg, seed, xtr[:, cc], try_, xva[:, cc])
                        total += loss(vy, pp).sum()
                    scores.append(total / len(dd))
                ix = int(np.argmin(scores))
                bestcfg = grid[ix]
                cvll = scores[ix]
                cc = select_columns(blocks, cols)
                p = fit_array(algo, bestcfg, seed, xt[:, cc], ty, xv[:, cc])
            models[algo, subset] = p
            candidates.append(
                dict(
                    algorithm=algo,
                    subset=subset,
                    blocks=blocks,
                    has_history=spec["has_history"],
                    additional_items=spec["additional_items"],
                    cv_loss=float(cvll),
                    config=bestcfg,
                )
            )
        candidates.append(
            dict(
                algorithm=algo,
                subset="P0",
                blocks=["H"],
                has_history=True,
                additional_items=0,
                cv_loss=float(cvbase["P0"]),
                config=None,
            )
        )
        models[algo, "P0"] = baseline["P0"]
    for algo in GRIDS:
        for history in [True, False]:
            benchmark = "P0" if history else "NONE"
            full = next((s["subset"] for s in FULLS if s["has_history"] == history))
            ref_loss = float(loss(y, baseline[benchmark]).mean())
            full_loss = float(loss(y, models[algo, full]).mean())
            for budget in BUDGETS:
                eligible = [
                    r
                    for r in candidates
                    if r["algorithm"] == algo
                    and r["additional_items"] <= budget
                    and (history or not r["has_history"])
                ]
                selected = min(
                    eligible, key=lambda r: (r["cv_loss"], r["additional_items"], r["subset"])
                )
                p = models[algo, selected["subset"]]
                ll = float(loss(y, p).mean())
                rows.append(
                    dict(
                        replicate=rep,
                        point_estimator=point,
                        algorithm=algo,
                        history_available=history,
                        budget=budget,
                        selected_subset=selected["subset"],
                        selected_config=json.dumps(selected["config"], sort_keys=True),
                        selected_items=selected["additional_items"],
                        cv_log_loss=selected["cv_loss"],
                        temporal_log_loss=ll,
                        benchmark=benchmark,
                        benchmark_log_loss=ref_loss,
                        full_subset=full,
                        full_log_loss=full_loss,
                        delta_vs_benchmark=ll - ref_loss,
                        delta_vs_full=ll - full_loss,
                        roc_auc=float(roc_auc_score(y, p)),
                        n_development=len(dd),
                        n_evaluation=len(ee),
                        unique_development=len(set(dev.id.iloc[dd])),
                        unique_evaluation=len(set(eva.id.iloc[ee])),
                    )
                )
    result = dict(replicate=rep, seconds=time.time() - start, rows=rows)
    (outdir / f"replicate_{rep:04d}.json").write_text(json.dumps(result, indent=2))
    return result


def run_context():
    return cache.context(ROOT, Path(__file__), GRIDS, 202610200)


def evaluate_guarded(rep, dev, eva, union, base_seed, outdir, point=False):
    context = run_context()
    result = evaluate(rep, dev, eva, union, base_seed, outdir, point)
    if run_context() != context:
        raise RuntimeError(
            "Bootstrap inputs or fitting dependencies changed during a replicate; no completion receipt committed."
        )
    cache.record_checkpoint(outdir / f"replicate_{rep:04d}.json", context)
    return result


def summarize(outdir, reps):
    context = run_context()
    paths = [
        outdir / f"replicate_{r:04d}.json"
        for r in range(reps + 1)
        if cache.receipt_path(outdir / f"replicate_{r:04d}.json").exists()
    ]
    for path in paths:
        cache.verify_checkpoint(path, context)
    results = [json.loads(path.read_text()) for path in paths]
    df = pd.DataFrame([r for result in results for r in result["rows"]])
    point = df[df.point_estimator]
    boot = df[~df.point_estimator]
    tables = ROOT / "revision/tables"
    audit = ROOT / "revision/audit"
    tables.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    point.to_csv(tables / "reselection_point_estimates.csv", index=False)
    boot.to_csv(tables / "reselection_bootstrap_replicates.csv", index=False)
    keys = ["algorithm", "history_available", "budget"]
    summ = []
    freq = []
    for key, g in boot.groupby(keys):
        p = point[
            (point.algorithm == key[0])
            & (point.history_available == key[1])
            & (point.budget == key[2])
        ].iloc[0]
        n = len(g)
        record = dict(zip(keys, key))
        record.update(
            replicates=n, point_subset=p.selected_subset, point_items=int(p.selected_items)
        )
        for col in ["temporal_log_loss", "delta_vs_benchmark", "delta_vs_full", "roc_auc"]:
            record[col + "_point"] = float(p[col])
            record[col + "_bootstrap_mean"] = float(g[col].mean())
            record[col + "_lower_025"] = float(g[col].quantile(0.025))
            record[col + "_upper_975"] = float(g[col].quantile(0.975))
            record[col + "_bootstrap_sd"] = float(g[col].std(ddof=1))
        record["fraction_delta_vs_benchmark_positive"] = float((g.delta_vs_benchmark > 0).mean())
        summ.append(record)
        for subset, h in g.groupby("selected_subset"):
            rate = len(h) / n
            freq.append(
                dict(
                    **dict(zip(keys, key)),
                    selected_subset=subset,
                    count=len(h),
                    frequency=rate,
                    monte_carlo_se=float(np.sqrt(rate * (1 - rate) / n)),
                )
            )
    pd.DataFrame(summ).to_csv(tables / "reselection_bootstrap_intervals.csv", index=False)
    pd.DataFrame(freq).to_csv(tables / "reselection_bootstrap_frequencies.csv", index=False)
    manifest = dict(
        status="complete" if len(boot) == reps * 8 else "in_progress",
        requested_replicates=reps,
        completed_replicates=int(boot.replicate.nunique()),
        point_estimator_replicate=0,
        bootstrap_scheme="multinomial resampling of participant union; common multiplicities across periods",
        selection_scope="All 34 portfolios eligible at budgets 1 and 4 plus P0; both algorithms and both information-availability conditions",
        full_references="Two 30-additional-item full models are independently retuned and refitted in every replicate",
        folds=5,
        duplicate_policy="All copies of a participant are in one validation fold; distinct sampled participants are stratified before copies are expanded",
        preprocessing="Median/mode imputation, numeric standardization and one-hot encoding fit afresh in each development training fold; no evaluation data in preprocessing or selection",
        inference="Pointwise percentile intervals for the complete focused-budget procedure, conditional on fixed wave design; no multiplicity-adjusted claim and no bootstrap p-value",
        original_grids=GRIDS,
        small_budget_scope_rationale="Reviewer requests specifically identify budget-1 power and budget-4 demographics harm; wider budgets are outside full-process bootstrap scope",
        seed_base=202610200,
        seconds=[r["seconds"] for r in results],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    manifest["run_fingerprint"] = run_context()["fingerprint"]
    manifest["cache_provenance"] = (
        "Input, fitting dependencies, grids, seed, environment and each checkpoint output are verified before reuse."
    )
    (audit / "reselection_bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    dev = pd.read_csv(ROOT / "private_data/development.csv", dtype={"id": str})
    eva = pd.read_csv(ROOT / "private_data/evaluation.csv", dtype={"id": str})
    union = sorted(set(dev.id) | set(eva.id))
    outdir = ROOT / "private_data/reselection_checkpoints"
    outdir.mkdir(exist_ok=True)
    cache.ensure_directory(outdir, run_context())
    with threadpool_limits(limits=1):
        if args.benchmark:
            r = evaluate_guarded(0, dev, eva, union, 202610200, outdir, True)
            print(json.dumps({"benchmark_seconds": r["seconds"], "rows": r["rows"]}, indent=2))
            return
        if not (outdir / "replicate_0000.json").exists():
            evaluate_guarded(0, dev, eva, union, 202610200, outdir, True)
        remaining = [
            r for r in range(1, args.reps + 1) if not (outdir / f"replicate_{r:04d}.json").exists()
        ]
        for i, result in enumerate(
            joblib.Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                (
                    joblib.delayed(evaluate_guarded)(r, dev, eva, union, 202610200, outdir)
                    for r in remaining
                )
            )
        ):
            print(
                f"replicate {result['replicate']} completed {result['seconds']:.1f}s", flush=True
            )
            if (i + 1) % 10 == 0:
                summarize(outdir, args.reps)
        print(json.dumps(summarize(outdir, args.reps), indent=2))


if __name__ == "__main__":
    main()
