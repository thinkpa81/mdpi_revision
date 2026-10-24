import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["algorithm", "history_available", "budget"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, out):
    tables = root / "revision/tables"
    out.mkdir(parents=True, exist_ok=True)
    filenames = [
        "round2_reselection_bootstrap_replicates.csv",
        "round2_reselection_point_estimates.csv",
        "round2_reselection_bootstrap_intervals.csv",
        "reselection_bootstrap_replicates.csv",
        "reselection_point_estimates.csv",
    ]
    boot, point, reported, old_boot, old_point = [pd.read_csv(tables / f) for f in filenames]
    assert len(boot) == 8000 and len(point) == 8
    assert set(boot.replicate) == set(range(1, 1001))
    assert not boot.duplicated(["replicate"] + KEYS).any()
    assert set(boot.budget) == {1, 4}
    assert set(boot.algorithm) == {"ridge", "hgb"}
    a = old_boot.sort_values(["replicate"] + KEYS).reset_index(drop=True)
    b = boot[boot.replicate <= 200].sort_values(["replicate"] + KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(
        old_point.sort_values(KEYS).reset_index(drop=True),
        point.sort_values(KEYS).reset_index(drop=True),
        check_exact=True,
    )
    rows = []
    max_reported_diff = 0.0
    for key, g in boot.groupby(KEYS, sort=True):
        assert len(g) == 1000
        p = point.set_index(KEYS).loc[key]
        r = reported.set_index(KEYS).loc[key]
        for reference in ["benchmark", "full"]:
            column = "delta_vs_" + reference
            v = g[column].to_numpy()
            t = float(p[column])
            lo, hi = np.quantile(v, [0.025, 0.975], method="linear")
            mean = float(v.mean())
            sd = float(v.std(ddof=1))
            basic_lo, basic_hi = (2 * t - hi, 2 * t - lo)
            for value, suffix in [
                (t, "point"),
                (mean, "bootstrap_mean"),
                (sd, "bootstrap_sd"),
                (lo, "lower_025"),
                (hi, "upper_975"),
            ]:
                max_reported_diff = max(
                    max_reported_diff, abs(value - float(r[column + "_" + suffix]))
                )
            assert np.isfinite(v).all()
            rows.append(
                dict(zip(KEYS, key))
                | {
                    "reference": reference,
                    "point_subset": p.selected_subset,
                    "replicates": len(g),
                    "point_estimate": t,
                    "bootstrap_mean": mean,
                    "bootstrap_bias_mean_minus_point": mean - t,
                    "bootstrap_sd": sd,
                    "percentile_lower_025": float(lo),
                    "percentile_upper_975": float(hi),
                    "basic_lower_025": float(basic_lo),
                    "basic_upper_975": float(basic_hi),
                    "percentile_contains_zero": bool(lo <= 0 <= hi),
                    "basic_contains_zero": bool(basic_lo <= 0 <= basic_hi),
                    "percentile_direction": (
                        "benefit" if hi < 0 else "harm" if lo > 0 else "includes_zero"
                    ),
                    "basic_direction": (
                        "benefit" if basic_hi < 0 else "harm" if basic_lo > 0 else "includes_zero"
                    ),
                    "positive_replicates": int((v > 0).sum()),
                    "negative_replicates": int((v < 0).sum()),
                    "zero_replicates": int((v == 0).sum()),
                    "positive_fraction_descriptive_only": float((v > 0).mean()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "interval_method_sensitivity.csv", index=False)
    summary = {}
    for ref, g in result.groupby("reference"):
        summary[ref] = {
            "procedures": len(g),
            "percentile_includes_zero": int(g.percentile_contains_zero.sum()),
            "basic_includes_zero": int(g.basic_contains_zero.sum()),
            "basic_excludes_zero_benefit": int((g.basic_direction == "benefit").sum()),
            "basic_excludes_zero_harm": int((g.basic_direction == "harm").sum()),
            "maximum_absolute_bootstrap_bias": float(
                g.bootstrap_bias_mean_minus_point.abs().max()
            ),
        }
    assert max_reported_diff < 1e-12
    audit = {
        "status": "passed",
        "analysis": "interval-construction sensitivity; original fits unchanged",
        "input_sha256": {f: digest(tables / f) for f in filenames},
        "records": 8000,
        "procedures_sharing_resamples": 8,
        "unique_resamples": 1000,
        "reference_types": 2,
        "intervals_reported": 16,
        "quantiles": "NumPy linear 0.025 and 0.975 quantiles",
        "basic_formula": "[2*point - q0.975, 2*point - q0.025]",
        "bootstrap_bias_definition": "bootstrap mean minus original-sample point estimate",
        "reported_summary_max_abs_difference": max_reported_diff,
        "first_200_replicates_exactly_match_prior_stage": True,
        "point_estimator_exactly_matches_prior_stage": True,
        "summary": summary,
        "interpretation": [
            "Percentile intervals remain the primary, originally specified full-process sensitivity summaries.",
            "Basic intervals are an additional diagnostic; disagreement indicates interval-construction sensitivity.",
            "A nonsignificant interval is not evidence of equality or absence of benefit.",
            "Neither method is simultaneous across procedures or guaranteed to resolve nonregular model-selection uncertainty.",
            "Positive bootstrap fractions are descriptive, not p-values or posterior harm probabilities.",
            "Full-model comparisons address performance relative to a refitted full model and do not by themselves establish collection value relative to P0/NONE.",
        ],
    }
    (out / "interval_method_sensitivity.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O or -OO so validation assertions remain enabled.")
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run(args.package, args.out or args.package / "revision/results")
