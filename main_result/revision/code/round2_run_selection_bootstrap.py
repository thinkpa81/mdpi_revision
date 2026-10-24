from pathlib import Path
import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import argparse, hashlib, json, shutil, time, sys, platform, fcntl
import numpy as np
import pandas as pd
import sklearn, joblib
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_selection_bootstrap as original
import bootstrap_cache as cache

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "private_data/round2_reselection_checkpoints"
TABLES = ROOT / "revision/tables"
AUDIT = ROOT / "revision/audit"
SEED = 202610200
KEYS = ["algorithm", "history_available", "budget"]
EXPECTED = {
    "development": "a3beced41c288b5dd4454598b26bde9007e78249ba597cd95a3b6c4234aea022",
    "evaluation": "24f0af7707c883fc9ebef22186ccfbc5811f05f056a40da883ae56396e1187ce",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_and_copy():
    source = ROOT / "private_data/reselection_checkpoints"
    manifest = json.loads((AUDIT / "reselection_bootstrap_manifest.json").read_text())
    validation = json.loads((AUDIT / "reselection_bootstrap_validation.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_replicates"] == 200
    assert validation["status"] == "passed" and validation["replicates"] == 200
    assert sha(Path(original.__file__)) == manifest["script_sha256"]
    run_context = original.run_context()
    assert manifest["run_fingerprint"] == run_context["fingerprint"]
    cache.ensure_directory(source, run_context)
    cache.ensure_directory(CHECKPOINTS, run_context)
    for name in ["development", "evaluation"]:
        assert (
            sha(ROOT / f"private_data/{name}.csv")
            == EXPECTED[name]
            == validation[f"input_{name}_sha256"]
        )
    assert manifest["seed_base"] == SEED and manifest["folds"] == 5
    assert manifest["original_grids"] == original.GRIDS
    CHECKPOINTS.mkdir(exist_ok=True)
    digests = {}
    for r in range(201):
        src = source / f"replicate_{r:04d}.json"
        dest = CHECKPOINTS / src.name
        payload = json.loads(src.read_text())
        assert payload["replicate"] == r and len(payload["rows"]) == 8
        assert all(
            (x["point_estimator"] == (r == 0) and x["replicate"] == r for x in payload["rows"])
        )
        cache.verify_checkpoint(src, run_context)
        if not dest.exists():
            shutil.copy2(src, dest)
            shutil.copy2(cache.receipt_path(src), cache.receipt_path(dest))
        cache.verify_checkpoint(dest, run_context)
        assert sha(dest) == sha(src)
        digests[src.name] = sha(src)
    provenance = dict(
        status="verified",
        reused_bootstrap_replicates=200,
        reused_point_estimator=True,
        source_evaluation_script_sha256=sha(Path(original.__file__)),
        model_engine_sha256=sha(ROOT / "revision/code/original_model_engine.py"),
        run_fingerprint=run_context["fingerprint"],
        seed_base=SEED,
        input_sha256={k: EXPECTED[k] for k in ["development", "evaluation"]},
        checkpoint_sha256=digests,
    )
    (AUDIT / "round2_reselection_reuse_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )


def summarize(reps):
    context = original.run_context()
    ids = [
        r
        for r in range(reps + 1)
        if cache.receipt_path(CHECKPOINTS / f"replicate_{r:04d}.json").exists()
    ]
    for r in ids:
        cache.verify_checkpoint(CHECKPOINTS / f"replicate_{r:04d}.json", context)
    results = [json.loads((CHECKPOINTS / f"replicate_{r:04d}.json").read_text()) for r in ids]
    for result in results:
        assert len(result["rows"]) == 8 and result["replicate"] in ids
    df = pd.DataFrame([row for result in results for row in result["rows"]])
    point = df[df.point_estimator].copy()
    boot = df[~df.point_estimator].copy()
    assert not df.duplicated(["replicate"] + KEYS).any()
    point.to_csv(TABLES / "round2_reselection_point_estimates.csv", index=False)
    boot.to_csv(TABLES / "round2_reselection_bootstrap_replicates.csv", index=False)
    summaries = []
    frequencies = []
    for key, g in boot.groupby(KEYS):
        p = point[
            (point.algorithm == key[0])
            & (point.history_available == key[1])
            & (point.budget == key[2])
        ].iloc[0]
        n = len(g)
        record = dict(zip(KEYS, key))
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
        summaries.append(record)
        for subset, h in g.groupby("selected_subset"):
            rate = len(h) / n
            frequencies.append(
                dict(
                    **dict(zip(KEYS, key)),
                    selected_subset=subset,
                    count=len(h),
                    frequency=rate,
                    monte_carlo_se=float(np.sqrt(rate * (1 - rate) / n)),
                )
            )
    pd.DataFrame(summaries).to_csv(
        TABLES / "round2_reselection_bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(frequencies).to_csv(
        TABLES / "round2_reselection_bootstrap_frequencies.csv", index=False
    )
    manifest = json.loads((AUDIT / "reselection_bootstrap_manifest.json").read_text())
    manifest.update(
        status="complete" if set(ids) == set(range(reps + 1)) else "in_progress",
        requested_replicates=reps,
        completed_replicates=int(boot.replicate.nunique()),
        original_replicates_reused=200,
        new_replicates_completed=sum((r > 200 for r in ids)),
        seconds=[r["seconds"] for r in results],
        script_sha256=sha(Path(__file__)),
        fitting_script_sha256=sha(Path(original.__file__)),
        checkpoint_directory="private_data/round2_reselection_checkpoints",
        package_versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            joblib=joblib.__version__,
        ),
        quantile_interpolation="linear (pandas default)",
        output_prefix="round2_",
        expected_observations_per_percentile_tail=0.025 * int(boot.replicate.nunique()),
    )
    (AUDIT / "round2_reselection_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-new", type=int)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    assert args.reps >= 200 and 1 <= args.jobs <= 8
    CHECKPOINTS.mkdir(exist_ok=True)
    with (CHECKPOINTS / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_and_copy()
        remaining = [
            r
            for r in range(201, args.reps + 1)
            if not (CHECKPOINTS / f"replicate_{r:04d}.json").exists()
        ]
        if args.max_new is not None:
            remaining = remaining[: args.max_new]
        if args.summarize_only:
            remaining = []
        dev = pd.read_csv(ROOT / "private_data/development.csv", dtype={"id": str})
        eva = pd.read_csv(ROOT / "private_data/evaluation.csv", dtype={"id": str})
        union = sorted(set(dev.id) | set(eva.id))
        start = time.time()
        print(
            json.dumps(
                dict(
                    event="start",
                    jobs=args.jobs,
                    remaining_scheduled=len(remaining),
                    requested=args.reps,
                )
            ),
            flush=True,
        )
        with threadpool_limits(limits=1):
            for i, result in enumerate(
                joblib.Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                    (
                        joblib.delayed(original.evaluate_guarded)(
                            r, dev, eva, union, SEED, CHECKPOINTS
                        )
                        for r in remaining
                    )
                )
            ):
                print(
                    json.dumps(
                        dict(
                            event="completed",
                            replicate=result["replicate"],
                            seconds=round(result["seconds"], 3),
                            session_completed=i + 1,
                            session_wall_seconds=round(time.time() - start, 3),
                        )
                    ),
                    flush=True,
                )
                if (i + 1) % 20 == 0:
                    summarize(args.reps)
        manifest = summarize(args.reps)
        print(
            json.dumps(
                dict(
                    event="session_end",
                    status=manifest["status"],
                    total_completed=manifest["completed_replicates"],
                    session_wall_seconds=time.time() - start,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
