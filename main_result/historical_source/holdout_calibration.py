from __future__ import annotations
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

BLOCKS = {
    "H": (["prior_any_use"], [], 0),
    "D": ([], ["gender", "age_group", "education", "region"], 4),
    "B": (["benefit_expectation"], [], 1),
    "M": (["data_management"], [], 1),
    "P": (["safeguard_expectation"], [], 11),
    "C": ([], ["consent_strategy"], 1),
    "A": (["genai_experience"], [], 1),
    "K": (["digital_competence_11"], [], 11),
}
OPTIONAL = ["D", "B", "M", "P", "C", "A", "K"]
NUM = [v for n, c, k in BLOCKS.values() for v in n]
CAT = [v for n, c, k in BLOCKS.values() for v in c]
RIDGE = [0.001, 0.01, 0.1, 1.0, 10.0]
HGB = [{"max_leaf_nodes": 3, "learning_rate": 0.05}, {"max_leaf_nodes": 7, "learning_rate": 0.05}]
SEED = 202610200


def loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def metrics(y, p):
    return {
        "n": len(y),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "log_loss": float(loss(y, p).mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def specs():
    out = []
    for h in (True, False):
        for mask in range(128):
            blocks = (["H"] if h else []) + [b for j, b in enumerate(OPTIONAL) if mask & 1 << j]
            cost = sum((BLOCKS[b][2] for b in blocks))
            if cost <= 4 or len(blocks) == 7 + int(h):
                out.append(
                    {
                        "subset": "_".join(blocks) or "NONE",
                        "blocks": blocks,
                        "has_history": h,
                        "cost": cost,
                    }
                )
    return out


def prepared(train, targets):
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CAT,
            ),
        ]
    )
    xt = prep.fit_transform(train)
    xx = [prep.transform(t) for t in targets]
    inds = {n: [j] for j, n in enumerate(NUM)}
    off = len(NUM)
    for n, cats in zip(CAT, prep.named_transformers_["cat"].named_steps["encode"].categories_):
        inds[n] = list(range(off, off + len(cats)))
        off += len(cats)
    blockidx = {b: sum([inds[n] for n in ns + cs], []) for b, (ns, cs, k) in BLOCKS.items()}
    indices = {
        s["subset"]: sum([inds[n] for b in s["blocks"] for n in BLOCKS[b][0]], [])
        + sum([inds[n] for b in s["blocks"] for n in BLOCKS[b][1]], [])
        for s in specs()
    }
    return (xt, xx, indices)


def fit_arr(xt, y, xv, algo, config):
    if xt.shape[1] == 0:
        return np.full(len(xv), (y.sum() + 1) / (len(y) + 2))
    if algo == "ridge":
        m = LogisticRegression(C=config, solver="lbfgs", max_iter=2000, random_state=20260907)
    else:
        m = HistGradientBoostingClassifier(
            **config,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260907,
        )
    m.fit(xt, y)
    return m.predict_proba(xv)[:, 1]


def p0(train, targets):
    return [
        t.prior_any_use.map(
            {
                h: (train.loc[train.prior_any_use == h, "outcome_any_use"].sum() + 1)
                / ((train.prior_any_use == h).sum() + 2)
                for h in (0, 1)
            }
        ).to_numpy()
        for t in targets
    ]


def one_repeat(dev, eva, rep):
    with threadpool_limits(limits=1):
        ids = np.arange(len(dev))
        tr, va = train_test_split(
            ids, test_size=0.3, stratify=dev.outcome_any_use, random_state=SEED + rep
        )
        train = dev.iloc[tr].reset_index(drop=True)
        valid = dev.iloc[va].reset_index(drop=True)
        assert set(train.id).isdisjoint(valid.id)
        y = train.outcome_any_use.to_numpy(dtype=int)
        folds = list(
            StratifiedKFold(5, shuffle=True, random_state=SEED + 1000 + rep).split(train, y)
        )
        preps = []
        for t, v in folds:
            xt, xx, ix = prepared(train.iloc[t], [train.iloc[v]])
            preps.append((t, v, xt, xx[0], ix))
        allspec = specs()
        chosen = []
        p0oof = np.zeros(len(train))
        for t, v in folds:
            p0oof[v] = p0(train.iloc[t], [train.iloc[v]])[0]
        for algo in ("ridge", "hgb"):
            grid = RIDGE if algo == "ridge" else HGB
            for s in allspec:
                scores = []
                for cfg in grid:
                    oof = np.zeros(len(train))
                    for t, v, xt, xv, ix in preps:
                        jj = ix[s["subset"]]
                        oof[v] = fit_arr(xt[:, jj], y[t], xv[:, jj], algo, cfg)
                    scores.append(float(loss(y, oof).mean()))
                k = int(np.argmin(scores))
                chosen.append(dict(s, algorithm=algo, config=grid[k], cv_loss=scores[k]))
            chosen.append(
                {
                    "subset": "P0",
                    "blocks": ["H"],
                    "has_history": True,
                    "cost": 0,
                    "algorithm": algo,
                    "config": None,
                    "cv_loss": float(loss(y, p0oof).mean()),
                }
            )
        xt, targets, ix = prepared(train, [valid, eva])
        rows = []
        sels = []
        cache = {}
        for algo in ("ridge", "hgb"):
            candidates = [s for s in chosen if s["algorithm"] == algo]
            selections = []
            for availability in ("history_available", "history_unavailable"):
                for budget in (1, 4):
                    eligible = [
                        s
                        for s in candidates
                        if s["cost"] <= budget
                        and (availability == "history_available" or not s["has_history"])
                    ]
                    s = min(eligible, key=lambda s: (s["cv_loss"], s["cost"], s["subset"]))
                    selections.append((availability, f"budget_{budget}", s))
            for subset in ["NONE", "P0", "D", "H_D", "D_B_M_P_C_A_K", "H_D_B_M_P_C_A_K"]:
                s = next((s for s in candidates if s["subset"] == subset))
                selections.append(("reference", subset, s))
            for avail, policy, s in selections:
                key = (algo, s["subset"])
                if key not in cache:
                    if s["subset"] == "P0":
                        cache[key] = p0(train, [valid, eva])
                    else:
                        jj = ix[s["subset"]]
                        comb = np.concatenate([t[:, jj] for t in targets])
                        pp = fit_arr(xt[:, jj], y, comb, algo, s["config"])
                        cache[key] = [pp[: len(valid)], pp[len(valid) :]]
                sels.append(
                    {
                        "repeat": rep,
                        "algorithm": algo,
                        "availability": avail,
                        "policy": policy,
                        "subset": s["subset"],
                        "config": json.dumps(s["config"]),
                        "additional_items": s["cost"],
                        "inner_cv_log_loss": s["cv_loss"],
                        "n_train": len(train),
                        "n_holdout": len(valid),
                    }
                )
                for sample, target, pp in zip(
                    ("within_period_holdout", "matched_training_temporal"),
                    (valid, eva),
                    cache[key],
                ):
                    bench = (
                        p0(train, [target])[0]
                        if avail == "history_available" or "H" in s["blocks"]
                        else np.full(len(target), (y.sum() + 1) / (len(y) + 2))
                    )
                    rows.append(
                        dict(
                            sels[-1],
                            sample=sample,
                            **metrics(target.outcome_any_use.to_numpy(), pp),
                            delta_log_loss_vs_benchmark=float(
                                (
                                    loss(target.outcome_any_use.to_numpy(), pp)
                                    - loss(target.outcome_any_use.to_numpy(), bench)
                                ).mean()
                            ),
                        )
                    )
        return (rows, sels)


def run_holdout(args):
    dev = pd.read_csv(args.data / "development.csv", dtype={"id": str})
    eva = pd.read_csv(args.data / "evaluation.csv", dtype={"id": str})
    allrows = []
    allsels = []
    start = time.time()
    for begin in range(0, args.repeats, args.jobs):
        r = Parallel(n_jobs=args.jobs)(
            (
                delayed(one_repeat)(dev, eva, rep)
                for rep in range(begin, min(args.repeats, begin + args.jobs))
            )
        )
        for rows, sels in r:
            allrows.extend(rows)
            allsels.extend(sels)
        pd.DataFrame(allrows).to_csv(
            args.out / "within_period_holdout_repetitions.csv", index=False
        )
        pd.DataFrame(allsels).to_csv(
            args.out / "within_period_holdout_selections.csv", index=False
        )
        print(
            f"holdout {len(set((s['repeat'] for s in allsels)))}/{args.repeats}; elapsed {time.time() - start:.1f}s",
            flush=True,
        )
    df = pd.DataFrame(allrows)
    keys = ["algorithm", "availability", "policy", "sample"]
    summary = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k))
        row["repeats"] = len(g)
        for col in ("log_loss", "roc_auc", "brier", "delta_log_loss_vs_benchmark"):
            row[col + "_mean"] = g[col].mean()
            row[col + "_split_sd"] = g[col].std()
            row[col + "_split_q025"] = g[col].quantile(0.025)
            row[col + "_split_q975"] = g[col].quantile(0.975)
        summary.append(row)
    pd.DataFrame(summary).to_csv(args.out / "within_period_holdout_summary.csv", index=False)
    (args.out.parent / "audit/within_period_protocol.json").write_text(
        json.dumps(
            {
                "status": "post-review sensitivity; temporal outcomes previously examined",
                "repeats": args.repeats,
                "seed": SEED,
                "outer_split": "stratified 70 percent train 30 percent holdout",
                "inner_cv": "5-fold stratified on outer training only",
                "algorithms": ["ridge", "hgb"],
                "ridge_grid": RIDGE,
                "hgb_grid": HGB,
                "candidate_space": "all original block subsets eligible for budgets 1 and 4 plus both full-information models; P0 included",
                "evaluation_selection_access": False,
                "matched_temporal": "same model trained in each holdout split evaluated on 2024 to 2025",
                "uncertainty": "split quantiles and SD describe split variability, not confidence intervals because repeats overlap",
                "training_sizes": sorted(set(df.n_train)),
            },
            indent=2,
        )
    )


def recal_fit(y, p, kind):
    z = logit(np.clip(p, 1e-08, 1 - 1e-08))
    if kind == "intercept":
        a = brentq(lambda a: float((expit(z + a) - y).sum()), -40.0, 40.0)
        return (a, 1.0)
    if np.std(z) < 1e-12:
        return (logit(np.clip(y.mean(), 1e-08, 1 - 1e-08)), 0.0)

    def fun(b):
        eta = b[0] + b[1] * z
        v = np.logaddexp(0, eta) - y * eta
        r = expit(eta) - y
        return (v.mean(), np.array([r.mean(), (r * z).mean()]))

    fit = minimize(
        fun, [0.0, 1.0], jac=True, method="BFGS", options={"gtol": 1e-09, "maxiter": 1000}
    )
    if not np.isfinite(fit.x).all():
        raise RuntimeError("calibration fit failed")
    return tuple(fit.x)


def recal_predict(p, a, b):
    return expit(a + b * logit(np.clip(p, 1e-08, 1 - 1e-08)))


def run_calibration(args):
    eva = pd.read_csv(args.data / "evaluation.csv", dtype={"id": str})
    arr = np.load(args.data / "refit_all_predictions.npz")
    prediction_ids = pd.read_csv(args.data / "refit_prediction_ids.csv", dtype={"id": str})
    assert (
        prediction_ids.id.tolist() == eva.id.tolist()
    ), "Prediction row order does not match evaluation IDs"
    print("available prediction keys", arr.files[:8], flush=True)
    wanted = ["P0"] + [
        a + "::" + s
        for a in ("ridge", "hgb")
        for s in ["NONE", "B", "H_B", "D", "H_D", "D_B_M_P_C_A_K", "H_D_B_M_P_C_A_K"]
    ]
    out = []
    bootrows = []
    nboot = args.calibration_bootstrap
    for key in wanted:
        if key not in arr.files:
            raise KeyError(f"Expected prediction key missing: {key}")
        for sample, mask in [
            ("whole", np.ones(len(eva), bool)),
            ("person_disjoint", eva.seen_development_id.to_numpy() == 0),
        ]:
            y = eva.outcome_any_use.to_numpy(dtype=int)[mask]
            p = arr[key][mask]
            raw = float(loss(y, p).mean())
            entropy = float(loss(y, np.full(len(y), y.mean())).mean())
            a1, b1 = recal_fit(y, p, "intercept")
            p1 = recal_predict(p, a1, b1)
            a2, b2 = recal_fit(y, p, "slope")
            p2 = recal_predict(p, a2, b2)
            pcf = np.empty(len(y))
            picf = np.empty(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 5000).split(y, y):
                a, b = recal_fit(y[tr], p[tr], "slope")
                pcf[va] = recal_predict(p[va], a, b)
                a, b = recal_fit(y[tr], p[tr], "intercept")
                picf[va] = recal_predict(p[va], a, b)
            r = {
                "source_model": key,
                "sample": sample,
                "n": len(y),
                "raw_log_loss": raw,
                "outcome_entropy": entropy,
                "intercept_only_apparent_log_loss": float(loss(y, p1).mean()),
                "intercept_slope_apparent_log_loss": float(loss(y, p2).mean()),
                "intercept_only_crossfit_log_loss": float(loss(y, picf).mean()),
                "intercept_slope_crossfit_log_loss": float(loss(y, pcf).mean()),
                "calibration_intercept": a2,
                "calibration_slope": b2,
                "offset_intercept": a1,
                "raw_auc": float(roc_auc_score(y, p)),
                "apparent_recal_auc": float(roc_auc_score(y, p2)),
            }
            r["affine_miscalibration_component"] = raw - r["intercept_slope_apparent_log_loss"]
            r["affine_resolution_component"] = entropy - r["intercept_slope_apparent_log_loss"]
            r["decomposition_residual"] = raw - (
                entropy - r["affine_resolution_component"] + r["affine_miscalibration_component"]
            )
            rng = np.random.default_rng(SEED + 6000)
            vals = []
            for ib in range(nboot):
                ix = rng.integers(0, len(y), len(y))
                yy = y[ix]
                pp = p[ix]
                a, b = recal_fit(yy, pp, "slope")
                lp = float(loss(yy, recal_predict(pp, a, b)).mean())
                lr = float(loss(yy, pp).mean())
                he = float(loss(yy, np.full(len(yy), yy.mean())).mean())
                vals.append((a, b, lr - lp, he - lp))
            vals = np.asarray(vals)
            for j, n in enumerate(
                [
                    "calibration_intercept",
                    "calibration_slope",
                    "affine_miscalibration_component",
                    "affine_resolution_component",
                ]
            ):
                r[n + "_ci_low"], r[n + "_ci_high"] = np.quantile(vals[:, j], [0.025, 0.975])
            r["bootstrap_replicates"] = nboot
            out.append(r)
    pd.DataFrame(out).to_csv(args.out / "calibration_decomposition.csv", index=False)
    (args.out.parent / "audit/calibration_protocol.json").write_text(
        json.dumps(
            {
                "status": "retrospective post-review diagnostic using evaluation labels",
                "identity": "raw log loss = outcome entropy - affine logistic resolution + affine logistic miscalibration",
                "scope": "This is an affine-score diagnostic; not a full nonparametric calibration/discrimination decomposition. AUC separately measures ranking. Negative slope can reverse ranking.",
                "apparent_recalibration": "intercept and slope fitted on the same evaluation observations; optimistic diagnostic",
                "crossfitted_recalibration": "5-fold stratified evaluation split; recalibrator fit on 4 folds and tested on fifth; base model fixed",
                "bootstrap": "paired evaluation-participant bootstrap; affine calibrator refitted in each draw; trained base prediction model remains fixed",
                "bootstrap_replicates": nboot,
                "caution": "Recalibration does not identify survey wording effects separately from time, sample composition, or behavior and is not prospective deployment validation.",
            },
            indent=2,
        )
    )
    print("calibration complete", len(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["holdout", "calibration"])
    ap.add_argument("--data", type=Path, default=Path("private_data"))
    ap.add_argument("--out", type=Path, default=Path("revision/tables"))
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--calibration-bootstrap", type=int, default=500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "holdout":
        run_holdout(args)
    else:
        run_calibration(args)


if __name__ == "__main__":
    main()
