import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model_sensitivity as ms

ROOT = ms.ROOT
P = ms.PRIVATE
T = ms.TABLES
B = 5000
SEED = 2026102017


def run():
    ms.verify_task_caches(require_complete=True)
    output = []
    family_rows = []
    heterogeneity = []
    for ip, pop in enumerate(["restricted", "full"]):
        dev = pd.read_csv(
            P / ("development.csv" if pop == "restricted" else "full_development.csv"),
            dtype={"id": str},
        )
        ev = pd.read_csv(
            P / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
            dtype={"id": str},
        )
        y = ev.outcome_any_use.to_numpy(int)
        pd0 = ms.orig.p0(dev, ev)
        pn = np.full(len(ev), (dev.outcome_any_use.sum() + 1) / (len(dev) + 2))
        losses = {"P0": ms.orig.loss_vector(y, pd0), "NONE": ms.orig.loss_vector(y, pn)}
        cols = []
        diffs = []
        designated = []
        for algo in ms.GRIDS:
            for rep in ms.REPS:
                arr = np.load(P / f"{pop}__{algo}__{rep}.npz", allow_pickle=True)
                p = arr["prediction"]
                assert np.array_equal(arr["evaluation_id"].astype(str), ev.id.to_numpy())
                ll = ms.orig.loss_vector(y, p)
                for benchmark in ["P0", "NONE"]:
                    idx = len(cols)
                    cols.append(
                        dict(
                            population=pop, algorithm=algo, representation=rep, benchmark=benchmark
                        )
                    )
                    diffs.append(ll - losses[benchmark])
                    if (
                        rep == "full_H"
                        and benchmark == "P0"
                        or (rep == "full_noH" and benchmark == "NONE")
                    ):
                        designated.append(idx)
        assert len(designated) == 8
        D = np.asarray(diffs).T
        groups = {"all": np.ones(len(ev), bool), "person_disjoint": ~ev.id.isin(dev.id).to_numpy()}
        if pop == "full":
            groups.update(
                hesitant=ev.hesitation_frequency.isin([1, 2]).to_numpy(),
                other=~ev.hesitation_frequency.isin([1, 2]).to_numpy(),
                low_or_no_hesitation=ev.hesitation_frequency.isin([3, 4]).to_numpy(),
                no_service_experience=(ev.hesitation_frequency == 9).to_numpy(),
            )
        group_boot = {}
        group_mean = {}
        for ig, (group, mask) in enumerate(groups.items()):
            d = D[mask]
            n = len(d)
            mean = d.mean(axis=0)
            rng = np.random.default_rng(SEED + ip * 100 + ig)
            boot = []
            with threadpool_limits(1):
                for start in range(0, B, 200):
                    w = rng.multinomial(n, np.full(n, 1 / n), size=min(200, B - start))
                    boot.append(w @ d / n)
            boot = np.concatenate(boot)
            group_boot[group] = boot
            group_mean[group] = mean
            low, high = np.quantile(boot, [0.025, 0.975], axis=0)
            centered = boot - mean
            critical = float(np.quantile(np.abs(centered[:, designated]).max(axis=1), 0.95))
            family_rows.append(
                dict(
                    population=pop,
                    evaluation_group=group,
                    n=n,
                    replicates=B,
                    family_size=8,
                    critical=critical,
                    family="4 algorithms x full_H vs P0 and full_noH vs NONE",
                    statistic="maximum absolute centered paired bootstrap mean loss difference",
                )
            )
            for j, meta in enumerate(cols):
                row = dict(
                    **meta,
                    evaluation_group=group,
                    n=n,
                    replicates=B,
                    delta_log_loss=mean[j],
                    pointwise_low=low[j],
                    pointwise_high=high[j],
                    bootstrap_se=boot[:, j].std(ddof=1),
                    in_full_model_family=j in designated,
                )
                row.update(
                    simultaneous_low=mean[j] - critical if j in designated else np.nan,
                    simultaneous_high=mean[j] + critical if j in designated else np.nan,
                )
                output.append(row)
        if pop == "full":
            diff = group_mean["hesitant"] - group_mean["other"]
            bootdiff = group_boot["hesitant"] - group_boot["other"]
            lo, hi = np.quantile(bootdiff, [0.025, 0.975], axis=0)
            crit = float(np.quantile(np.abs((bootdiff - diff)[:, designated]).max(axis=1), 0.95))
            for j, meta in enumerate(cols):
                heterogeneity.append(
                    dict(
                        **meta,
                        contrast="delta_hesitant_minus_delta_other",
                        difference=diff[j],
                        pointwise_low=lo[j],
                        pointwise_high=hi[j],
                        in_full_model_family=j in designated,
                        simultaneous_low=diff[j] - crit if j in designated else np.nan,
                        simultaneous_high=diff[j] + crit if j in designated else np.nan,
                        replicates=B,
                    )
                )
    pd.DataFrame(heterogeneity).to_csv(
        T / "model_sensitivity_hesitancy_heterogeneity.csv", index=False
    )
    df = pd.DataFrame(output)
    df.to_csv(T / "model_sensitivity_paired_bootstrap.csv", index=False)
    df.loc[df.in_full_model_family].to_csv(
        T / "model_sensitivity_full_model_intervals.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(T / "model_sensitivity_inference_families.csv", index=False)
    audit = dict(
        replicates=B,
        seed=SEED,
        conditional_predictions=True,
        retraining=False,
        selection_repeated=False,
        pointwise_interval="percentile bootstrap 95 percent",
        simultaneous_interval="observed mean plus or minus 95th percentile of maximum absolute centered bootstrap deviations",
        family="8 full model contrasts within each population and evaluation group; family comprises all four algorithms and both history conditions",
        sign="positive=model log loss greater than reference, worse prediction",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (ms.AUDIT / "model_sensitivity_intervals.json").write_text(json.dumps(audit, indent=2))
    print(
        df.query('evaluation_group=="all" and in_full_model_family')[
            [
                "population",
                "algorithm",
                "representation",
                "benchmark",
                "delta_log_loss",
                "pointwise_low",
                "pointwise_high",
                "simultaneous_low",
                "simultaneous_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
