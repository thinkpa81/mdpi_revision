from pathlib import Path
import ast, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    sums = json.loads((ROOT / "SHA256SUMS.json").read_text())
    for name, wanted in sums.items():
        p = ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != wanted:
            raise RuntimeError("File changed: " + name)
    syntax = 0
    for p in ROOT.rglob("*.py"):
        if any((x in p.parts for x in ["private_data", "private_raw"])):
            continue
        ast.parse(p.read_text(encoding="utf-8"))
        syntax += 1
    t = ROOT / "revision/tables"
    k = ["algorithm", "history_available", "budget"]
    b = pd.read_csv(t / "round2_reselection_bootstrap_replicates.csv")
    old = pd.read_csv(t / "reselection_bootstrap_replicates.csv")
    point = pd.read_csv(t / "round2_reselection_point_estimates.csv").set_index(k)
    report = pd.read_csv(t / "round2_reselection_bootstrap_intervals.csv").set_index(k)
    freq = pd.read_csv(t / "round2_reselection_bootstrap_frequencies.csv")
    assert len(b) == 8000 and (not b.duplicated(["replicate"] + k).any())
    assert set(b.replicate) == set(range(1, 1001))
    pd.testing.assert_frame_equal(
        b[b.replicate <= 200].reset_index(drop=True), old.reset_index(drop=True)
    )
    err = 0.0
    for key, g in b.groupby(k):
        assert len(g) == 1000
        for metric in ["temporal_log_loss", "delta_vs_benchmark", "delta_vs_full", "roc_auc"]:
            v = g[metric].to_numpy()
            for suffix, value in [
                ("bootstrap_mean", v.mean()),
                ("bootstrap_sd", v.std(ddof=1)),
                ("lower_025", np.quantile(v, 0.025)),
                ("upper_975", np.quantile(v, 0.975)),
            ]:
                e = abs(float(report.loc[key, metric + "_" + suffix]) - value)
                err = max(err, e)
                assert e < 1e-12
        mask = np.ones(len(freq), bool)
        for name, value in zip(k, key):
            mask &= freq[name].eq(value)
        assert (
            g.selected_subset.value_counts().to_dict()
            == freq[mask].set_index("selected_subset")["count"].to_dict()
        )
    diag = ROOT / "revision/results/interval_method_sensitivity.csv"
    n = 0
    if diag.exists():
        d = pd.read_csv(diag).set_index(k + ["reference"])
        assert len(d) == 16
        for key, g in b.groupby(k):
            for reference in ["benchmark", "full"]:
                metric = "delta_vs_" + reference
                v = g[metric].to_numpy()
                p = float(point.loc[key, metric])
                lo, hi = np.quantile(v, [0.025, 0.975])
                row = d.loc[*key, reference]
                for name, value in [
                    ("point_estimate", p),
                    ("bootstrap_mean", v.mean()),
                    ("bootstrap_bias_mean_minus_point", v.mean() - p),
                    ("percentile_lower_025", lo),
                    ("percentile_upper_975", hi),
                    ("basic_lower_025", 2 * p - hi),
                    ("basic_upper_975", 2 * p - lo),
                ]:
                    e = abs(float(row[name]) - value)
                    err = max(err, e)
                    assert e < 1e-12
                n += 1
    print(
        json.dumps(
            {
                "status": "passed",
                "file_hashes": len(sums),
                "python_files": syntax,
                "bootstrap_records": len(b),
                "procedures": 8,
                "replicates_per_procedure": 1000,
                "interval_method_contrasts": n,
                "first_200_preserved": True,
                "maximum_numeric_error": err,
                "scope": "File integrity and aggregate numerical reconstruction; no model refitting.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
