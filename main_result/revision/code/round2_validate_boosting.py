from __future__ import annotations
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import argparse, ast, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "private_data"
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
PARTS = P / "round2_boosting_parts"
PREDS = P / "round2_boosting_predictions"
MANIFEST = A / "round2_boosting_manifest.json"
RECEIPT = A / "round2_boosting_validation.json"
SOURCE = ROOT / "revision/code/round2_boosting.py"
SEED = 20261020
BOOT_SEED = 2026102031
B = 5000
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
MAXERR = {}
COUNTS = {}
HASHES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, label):
    if not bool(ok):
        raise AssertionError(label)


def close(actual, expected, label, tol=1e-10):
    aa = np.asarray(actual, dtype=float)
    bb = np.asarray(expected, dtype=float)
    require(aa.shape == bb.shape, label + ": shape")
    require(np.array_equal(np.isnan(aa), np.isnan(bb)), label + ": NaN mask")
    good = np.isfinite(aa) & np.isfinite(bb)
    require(np.array_equal(np.isfinite(aa), np.isfinite(bb)), label + ": finite mask")
    err = float(np.max(abs(aa[good] - bb[good]))) if good.any() else 0.0
    MAXERR[label] = max(MAXERR.get(label, 0.0), err)
    require(err <= tol, f"{label}: max absolute error {err:.6g} > {tol}")


def loss(y, p):
    q = np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    return -(y * np.log(q) + (1 - y) * np.log1p(-q))


def benchmark(dev, ev):
    y = dev.outcome_any_use.to_numpy(int)
    rates = {
        v: (y[dev.prior_any_use.eq(v)].sum() + 1) / (dev.prior_any_use.eq(v).sum() + 2)
        for v in [0, 1]
    }
    return {
        "P0": ev.prior_any_use.map(rates).to_numpy(float),
        "NONE": np.full(len(ev), (y.sum() + 1) / (len(dev) + 2)),
    }


def masks(dev, ev):
    out = {"all": np.ones(len(ev), bool), "person_disjoint": ~ev.id.isin(dev.id).to_numpy()}
    if "hesitation_frequency" in ev:
        h = ev.hesitation_frequency
        out.update(
            hesitant=h.isin([1, 2]).to_numpy(),
            other=h.isin([3, 4, 9]).to_numpy(),
            low_or_no_hesitation=h.isin([3, 4]).to_numpy(),
            no_service_experience=h.eq(9).to_numpy(),
        )
        out = {k: v for k, v in out.items() if v.sum() > 1}
    return out


def metrics(y, p):
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    m = len(pos)
    n = len(neg)
    if m >= 2 and n >= 2:
        vp = (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")) / (2 * n)
        vn = 1 - (np.searchsorted(pos, neg, "left") + np.searchsorted(pos, neg, "right")) / (2 * m)
        auc = vp.mean()
        se = np.sqrt(vp.var(ddof=1) / m + vn.var(ddof=1) / n)
        lo = max(0.0, auc - 1.95996398454 * se)
        hi = min(1.0, auc + 1.95996398454 * se)
    else:
        auc = lo = hi = np.nan
    yy = np.column_stack([1 - y, y]).astype(p.dtype)
    pp = np.column_stack([1 - p, p])
    brier = float(np.mean(np.sum((yy - pp) ** 2, axis=1)) * 0.5)
    return dict(
        n=len(y),
        events=int(y.sum()),
        prevalence=y.mean(),
        log_loss=loss(y, p).mean(),
        brier=brier,
        roc_auc=auc,
        auc_ci_low=lo,
        auc_ci_high=hi,
        average_precision=average_precision_score(y, p),
    )


def sameframe(actual, expected, keys, label):
    require(set(actual.columns) == set(expected.columns), label + ": columns")
    aa = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    bb = expected.sort_values(keys).reset_index(drop=True)
    require(len(aa) == len(bb), label + ": rows")
    for col in bb:
        if pd.api.types.is_numeric_dtype(bb[col]):
            close(aa[col], bb[col], label + ": " + col)
        else:
            require(
                aa[col]
                .fillna("<missing>")
                .astype(str)
                .equals(bb[col].fillna("<missing>").astype(str)),
                label + ": " + col,
            )


def check_source_and_manifest(man):
    require(man["status"] == "completed_all", "manifest must be completed_all")
    require(man["seed"] == SEED and man["folds"] == 5, "development fold specification")
    require(
        man["representations"] == REPS
        and man["grids"] == GRIDS
        and (man["fixed_parameters"] == FIXED),
        "declared grids/fixed settings",
    )
    require(
        man["expected_final_models"] == 48
        and man["expected_cv_fits"] == 2880
        and (man["expected_total_fits"] == 2928),
        "fit-count manifest",
    )
    require(man["evaluation_used_for_tuning"] is False, "development-only selection declaration")
    require(
        man["early_stopping"] is False
        and man["class_weights"] is False
        and (man["oversampling"] is False),
        "sampling/early stopping declarations",
    )
    require(
        man["inference"]["replicates"] == B and man["inference"]["seed"] == BOOT_SEED,
        "bootstrap manifest",
    )
    require(man["code_sha256"] == sha(SOURCE), "analysis-code hash")
    tree = ast.parse(SOURCE.read_text())
    env = {"__builtins__": {}, "dict": dict, "SEED": SEED}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [x.id for x in node.targets if isinstance(x, ast.Name)]
            for name in targets:
                if name in ["SEED", "B", "BOOT_SEED", "GRIDS", "FIXED"]:
                    got = eval(compile(ast.Expression(node.value), str(SOURCE), "eval"), env)
                    expected = {
                        "SEED": SEED,
                        "B": B,
                        "BOOT_SEED": BOOT_SEED,
                        "GRIDS": GRIDS,
                        "FIXED": FIXED,
                    }[name]
                    require(got == expected, "source constant " + name)
    for f, h in man["input_sha256"].items():
        require(sha(P / f) == h, "source data hash " + f)
        HASHES["private_source:" + f] = h
    HASHES["analysis_code"] = sha(SOURCE)


def validate():
    man = json.loads(MANIFEST.read_text())
    check_source_and_manifest(man)
    aggregate_metrics = pd.read_csv(T / "round2_boosting_metrics.csv")
    aggregate_tuning = pd.read_csv(T / "round2_boosting_tuning.csv")
    paired = pd.read_csv(T / "round2_boosting_paired_bootstrap.csv")
    full = pd.read_csv(T / "round2_boosting_full_model_intervals.csv")
    families = pd.read_csv(T / "round2_boosting_inference_families.csv")
    cached = {}
    parts_metrics = []
    parts_tuning = []
    verified_metric_rows = 0
    for pop in ["restricted", "full"]:
        dv = "development.csv" if pop == "restricted" else "full_development.csv"
        ee = "evaluation.csv" if pop == "restricted" else "full_evaluation.csv"
        dev = pd.read_csv(P / dv, dtype={"id": str})
        ev = pd.read_csv(P / ee, dtype={"id": str})
        require(dev.id.is_unique and ev.id.is_unique, pop + ": unique IDs")
        require(
            (len(dev), len(ev)) == ((1565, 1520) if pop == "restricted" else (3895, 3741)),
            pop + ": source sample sizes",
        )
        yd = dev.outcome_any_use.to_numpy(int)
        ye = ev.outcome_any_use.to_numpy(int)
        folds = list(
            StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(dev)), yd)
        )
        fold_n = np.array([len(va) for _, va in folds])
        group = masks(dev, ev)
        predictions = {}
        for algo in GRIDS:
            for rep, blocks in REPS.items():
                key = f"{pop}__{algo}__{rep}"
                f = PREDS / (key + ".npz")
                with np.load(f, allow_pickle=True) as ar:
                    require(
                        set(ar.files)
                        == {"prediction", "development_oof", "evaluation_id", "development_id"},
                        key + ": prediction array fields",
                    )
                    require(
                        np.array_equal(ar["evaluation_id"].astype(str), ev.id.to_numpy()),
                        key + ": evaluation ID order",
                    )
                    require(
                        np.array_equal(ar["development_id"].astype(str), dev.id.to_numpy()),
                        key + ": development ID order",
                    )
                    pred = ar["prediction"].copy()
                    oof = ar["development_oof"].copy()
                for v, n, label in [(pred, len(ev), "prediction"), (oof, len(dev), "OOF")]:
                    require(
                        v.shape == (n,) and np.isfinite(v).all() and ((v > 0) & (v < 1)).all(),
                        key + ": " + label + " validity",
                    )
                tuning = pd.read_csv(PARTS / (key + "_tuning.csv"))
                met = pd.read_csv(PARTS / (key + "_metrics.csv"))
                params = json.loads((PARTS / (key + "_params.json")).read_text())
                require(
                    len(tuning) == 12 and tuning.configuration_index.tolist() == list(range(12)),
                    key + ": 12 complete ordered configurations",
                )
                for col, val in [
                    ("population", pop),
                    ("algorithm", algo),
                    ("representation", rep),
                ]:
                    require(tuning[col].eq(val).all() and met[col].eq(val).all(), key + ": " + col)
                require(
                    [json.loads(v) for v in tuning.params] == GRIDS[algo],
                    key + ": actual configuration grid",
                )
                fl = np.array([json.loads(v) for v in tuning.fold_log_losses])
                require(
                    fl.shape == (12, 5) and np.isfinite(fl).all(), key + ": 60 fold loss receipts"
                )
                close(
                    tuning.development_cv_log_loss,
                    fl @ fold_n / fold_n.sum(),
                    "pooled CV from all fold-loss receipts",
                )
                chosen = int(np.argmin(tuning.development_cv_log_loss.to_numpy()))
                require(
                    tuning.selected.sum() == 1 and bool(tuning.selected.iloc[chosen]),
                    key + ": pooled-loss argmin/first tie",
                )
                require(
                    met.selected_configuration.eq(chosen).all(),
                    key + ": metric chosen configuration",
                )
                require(
                    all((json.loads(v) == GRIDS[algo][chosen] for v in met.selected_params)),
                    key + ": selected parameters",
                )
                close(
                    met.development_cv_log_loss,
                    np.full(len(met), tuning.development_cv_log_loss.iloc[chosen]),
                    "selected CV metric receipt",
                )
                close(
                    np.array([loss(yd[va], oof[va]).mean() for _, va in folds]),
                    fl[chosen],
                    "selected OOF fold losses",
                )
                close(
                    loss(yd, oof).mean(),
                    tuning.development_cv_log_loss.iloc[chosen],
                    "selected OOF pooled loss",
                )
                for k, v in {**FIXED[algo], **GRIDS[algo][chosen]}.items():
                    require(k in params and params[k] == v, key + ": fitted parameter " + k)
                require(
                    params.get("early_stopping_rounds") in [None, 0], key + ": no early stopping"
                )
                require(
                    params.get("class_weight") is None
                    and params.get("class_weights") is None
                    and (params.get("auto_class_weights") in [None, "None"]),
                    key + ": no class weights",
                )
                if algo == "catboost":
                    cats = (
                        ["gender", "age_group", "education", "region"] if "D" in blocks else []
                    ) + (["consent_strategy"] if "C" in blocks else [])
                    require(
                        params.get("cat_features") == cats, key + ": native categorical features"
                    )
                require(
                    set(met.evaluation_group) == set(group)
                    and (not met.evaluation_group.duplicated().any()),
                    key + ": evaluation groups",
                )
                for g, mask in group.items():
                    r = met[met.evaluation_group.eq(g)].iloc[0]
                    vals = metrics(ye[mask], pred[mask])
                    for col, v in vals.items():
                        close(float(r[col]), float(v), "metric " + col)
                    verified_metric_rows += 1
                HASHES["prediction:" + key] = sha(f)
                predictions[algo, rep] = pred
                parts_metrics.append(met)
                parts_tuning.append(tuning)
        cached[pop] = (dev, ev, group, predictions)
    sameframe(
        aggregate_metrics,
        pd.concat(parts_metrics, ignore_index=True),
        ["population", "algorithm", "representation", "evaluation_group"],
        "aggregate metrics",
    )
    sameframe(
        aggregate_tuning,
        pd.concat(parts_tuning, ignore_index=True),
        ["population", "algorithm", "representation", "configuration_index"],
        "aggregate tuning",
    )
    require(
        len(aggregate_tuning) == 576 and aggregate_tuning.selected.sum() == 48,
        "48 models / 576 grid rows",
    )
    COUNTS.update(
        selected_models=48,
        configuration_rows=576,
        verified_fold_loss_receipts=2880,
        final_fit_parameter_receipts=48,
        metric_rows=verified_metric_rows,
    )
    expected_rows = []
    expected_families = []
    for ip, pop in enumerate(["restricted", "full"]):
        dev, ev, group, preds = cached[pop]
        y = ev.outcome_any_use.to_numpy(int)
        refs = benchmark(dev, ev)
        rl = {k: loss(y, p) for k, p in refs.items()}
        cols = []
        ds = []
        designated = []
        for algo in GRIDS:
            for rep in REPS:
                ll = loss(y, preds[algo, rep])
                for ref in ["P0", "NONE"]:
                    j = len(cols)
                    cols.append(
                        dict(population=pop, algorithm=algo, representation=rep, benchmark=ref)
                    )
                    ds.append(ll - rl[ref])
                    if (rep, ref) in [("full_H", "P0"), ("full_noH", "NONE")]:
                        designated.append(j)
        require(len(designated) == 6, pop + ": six-contrast family")
        D = np.column_stack(ds)
        for ig, (g, mask) in enumerate(group.items()):
            d = D[mask]
            n = len(d)
            mu = d.mean(0)
            rng = np.random.default_rng(BOOT_SEED + 100 * ip + ig)
            chunks = []
            with threadpool_limits(1):
                for start in range(0, B, 200):
                    w = rng.multinomial(n, np.full(n, 1 / n), size=min(200, B - start))
                    chunks.append(w @ d / n)
            boot = np.concatenate(chunks)
            low, high = np.quantile(boot, [0.025, 0.975], axis=0)
            q = np.quantile(np.abs((boot - mu)[:, designated]).max(axis=1), 0.95)
            expected_families.append(
                dict(
                    population=pop,
                    evaluation_group=g,
                    n=n,
                    replicates=B,
                    family_size=6,
                    critical=q,
                    family="3 added algorithms x [full_H vs P0, full_noH vs NONE]",
                    statistic="maximum absolute centered paired bootstrap mean loss difference",
                )
            )
            for j, meta in enumerate(cols):
                selected = j in designated
                expected_rows.append(
                    dict(
                        **meta,
                        evaluation_group=g,
                        n=n,
                        replicates=B,
                        delta_log_loss=mu[j],
                        pointwise_low=low[j],
                        pointwise_high=high[j],
                        bootstrap_se=boot[:, j].std(ddof=1),
                        in_full_model_family=selected,
                        simultaneous_low=mu[j] - q if selected else np.nan,
                        simultaneous_high=mu[j] + q if selected else np.nan,
                    )
                )
            print("Verified bootstrap:", pop, g, "n=", n, flush=True)
    e = pd.DataFrame(expected_rows)
    ef = pd.DataFrame(expected_families)
    keys = ["population", "algorithm", "representation", "benchmark", "evaluation_group"]
    sameframe(paired, e, keys, "paired bootstrap")
    sameframe(full, e[e.in_full_model_family], keys, "full model bootstrap")
    sameframe(families, ef, ["population", "evaluation_group"], "bootstrap families")
    core = full[full.evaluation_group.isin(["all", "person_disjoint"])]
    require(len(core) == 24, "24 whole/person-disjoint full-model contrasts")
    require(
        (full.simultaneous_low <= full.delta_log_loss).all()
        and (full.delta_log_loss <= full.simultaneous_high).all(),
        "simultaneous interval orientation",
    )
    COUNTS.update(
        paired_bootstrap_rows=len(paired),
        full_model_interval_rows=len(full),
        whole_disjoint_full_model_rows=24,
        bootstrap_families=len(families),
        bootstrap_replicates_per_family=B,
    )
    for name in [
        "round2_boosting_metrics.csv",
        "round2_boosting_tuning.csv",
        "round2_boosting_paired_bootstrap.csv",
        "round2_boosting_full_model_intervals.csv",
        "round2_boosting_inference_families.csv",
    ]:
        HASHES[name] = sha(T / name)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-seconds", type=int, default=0)
    args = ap.parse_args()
    start = time.time()
    deadline = start + args.wait_seconds
    while True:
        status = json.loads(MANIFEST.read_text()).get("status") if MANIFEST.exists() else "missing"
        if status == "completed_all":
            break
        if time.time() >= deadline:
            print("NOT READY: manifest status is", status, "(requires completed_all)", flush=True)
            return 2
        time.sleep(min(5, max(0, deadline - time.time())))
    report = {
        "status": "failed",
        "scope": "Independent checks without refitting models or importing boosting libraries",
        "completion_gate": "completed_all",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "validator_sha256": sha(__file__),
    }
    try:
        man = validate()
        report["status"] = "passed"
        report["manifest_status"] = man["status"]
        report["limitations"] = [
            "Unselected configurations have no saved OOF vectors: their pooled scores are checked against five fold-loss receipts, not reconstructed from predictions.",
            "Saved selected configurations, OOF predictions, final parameter receipts, and outputs are verified; models are not retrained.",
        ]
    except Exception as exc:
        report["error"] = str(exc)
        report["exception_type"] = type(exc).__name__
        traceback.print_exc()
    report.update(
        counts=COUNTS,
        max_absolute_errors=MAXERR,
        sha256=HASHES,
        elapsed_seconds=time.time() - start,
    )
    RECEIPT.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(report["status"], str(RECEIPT), COUNTS, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
