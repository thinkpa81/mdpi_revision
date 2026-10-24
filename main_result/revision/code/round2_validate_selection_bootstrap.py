from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
KEYS = ["algorithm", "history_available", "budget"]
N = 1000


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = pd.read_csv(T / "round2_reselection_bootstrap_replicates.csv")
    f = pd.read_csv(T / "round2_reselection_bootstrap_frequencies.csv")
    q = pd.read_csv(T / "round2_reselection_bootstrap_intervals.csv")
    pt = pd.read_csv(T / "round2_reselection_point_estimates.csv")
    original = pd.read_csv(T / "reselection_bootstrap_replicates.csv")
    original_validation = json.loads((A / "reselection_bootstrap_validation.json").read_text())
    assert (
        sha(T / "reselection_bootstrap_replicates.csv")
        == original_validation["bootstrap_file_sha256"]
    )
    man = json.loads((A / "round2_reselection_bootstrap_manifest.json").read_text())
    assert man["status"] == "complete" and man["completed_replicates"] == N
    provenance = json.loads((A / "round2_reselection_reuse_provenance.json").read_text())
    for name in ["development", "evaluation"]:
        assert sha(ROOT / f"private_data/{name}.csv") == provenance["input_sha256"][name]
    assert (
        sha(ROOT / "revision/code/run_selection_bootstrap.py")
        == provenance["source_evaluation_script_sha256"]
        == man["fitting_script_sha256"]
    )
    assert (
        sha(ROOT / "revision/code/original_model_engine.py") == provenance["model_engine_sha256"]
    )
    assert (
        len(r) == N * 8
        and r.replicate.nunique() == N
        and (set(r.replicate) == set(range(1, N + 1)))
    )
    assert r.groupby(KEYS).size().eq(N).all() and len(r.groupby(KEYS)) == 8
    assert not r.duplicated(["replicate"] + KEYS).any()
    assert r.selected_items.le(r.budget).all()
    assert (
        r[r.history_available == False].selected_subset.str.contains("H|P0", regex=True).sum() == 0
    )
    assert np.allclose(
        r.temporal_log_loss - r.benchmark_log_loss, r.delta_vs_benchmark, atol=1e-12
    )
    assert np.allclose(r.temporal_log_loss - r.full_log_loss, r.delta_vs_full, atol=1e-12)
    assert np.allclose(f.groupby(KEYS).frequency.sum(), 1, atol=1e-12)
    assert f.groupby(KEYS)["count"].sum().eq(N).all()
    assert np.isfinite(r.select_dtypes("number").to_numpy()).all()
    assert r.roc_auc.between(0, 1).all() and r.temporal_log_loss.ge(0).all()
    assert (r.unique_development < r.n_development).all() and (
        r.unique_evaluation < r.n_evaluation
    ).all()
    comparecols = list(original.columns)
    pd.testing.assert_frame_equal(
        r[r.replicate <= 200][comparecols]
        .sort_values(["replicate"] + KEYS)
        .reset_index(drop=True),
        original.sort_values(["replicate"] + KEYS).reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        pt, pd.read_csv(T / "reselection_point_estimates.csv"), check_exact=True
    )
    rows = []
    diagnostics = []
    for key, g in r.groupby(KEYS):
        p = pt[
            (pt.algorithm == key[0]) & (pt.history_available == key[1]) & (pt.budget == key[2])
        ].iloc[0]
        s = q[
            (q.algorithm == key[0]) & (q.history_available == key[1]) & (q.budget == key[2])
        ].iloc[0]
        freq = f[(f.algorithm == key[0]) & (f.history_available == key[1]) & (f.budget == key[2])]
        expected = g.selected_subset.value_counts().sort_index()
        assert (
            expected.to_dict() == freq.set_index("selected_subset")["count"].sort_index().to_dict()
        )
        assert np.allclose(freq.frequency * (1 - freq.frequency) / N, freq.monte_carlo_se**2)
        for col in ["temporal_log_loss", "delta_vs_benchmark", "delta_vs_full", "roc_auc"]:
            assert np.isclose(s[col + "_point"], p[col], atol=1e-14)
            assert np.isclose(s[col + "_lower_025"], g[col].quantile(0.025), atol=1e-14)
            assert np.isclose(s[col + "_upper_975"], g[col].quantile(0.975), atol=1e-14)
        for k in [200, 400, 600, 800, 1000]:
            subset = g[g.replicate <= k]
            for contrast in ["delta_vs_benchmark", "delta_vs_full"]:
                rows.append(
                    dict(
                        **dict(zip(KEYS, key)),
                        replicates=k,
                        contrast=contrast,
                        lower_025=subset[contrast].quantile(0.025),
                        upper_975=subset[contrast].quantile(0.975),
                        original_selection_recurrence=float(
                            (subset.selected_subset == p.selected_subset).mean()
                        ),
                        original_selection=p.selected_subset,
                    )
                )
        for b in range(5):
            subset = g[g.replicate.between(b * 200 + 1, (b + 1) * 200)]
            for contrast in ["delta_vs_benchmark", "delta_vs_full"]:
                diagnostics.append(
                    dict(
                        **dict(zip(KEYS, key)),
                        batch=b + 1,
                        replicate_start=b * 200 + 1,
                        replicate_end=(b + 1) * 200,
                        contrast=contrast,
                        lower_025=subset[contrast].quantile(0.025),
                        upper_975=subset[contrast].quantile(0.975),
                        original_selection_recurrence=float(
                            (subset.selected_subset == p.selected_subset).mean()
                        ),
                    )
                )
    pd.DataFrame(rows).to_csv(T / "round2_reselection_monte_carlo_convergence.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(T / "round2_reselection_monte_carlo_batches.csv", index=False)
    checks = dict(
        status="passed",
        input_data_and_fitting_code_unchanged=True,
        replicates=N,
        procedures=8,
        total_rows=len(r),
        original_200_rows_preserved_exactly=True,
        point_estimator_preserved_exactly=True,
        portfolio_frequency_sums_one=True,
        budget_constraints_satisfied=True,
        history_unavailable_excludes_H_and_P0=True,
        reported_differences_recomputed=True,
        reported_intervals_recomputed=True,
        all_replicates_have_duplicate_participants=True,
        duplicate_group_disjoint_assertions="Executed inside every development CV split",
        input_development_sha256=sha(ROOT / "private_data/development.csv"),
        input_evaluation_sha256=sha(ROOT / "private_data/evaluation.csv"),
        maximum_frequency_monte_carlo_se=float(f.monte_carlo_se.max()),
        maximum_possible_frequency_monte_carlo_se=float(np.sqrt(0.25 / N)),
        bootstrap_file_sha256=sha(T / "round2_reselection_bootstrap_replicates.csv"),
        percentile_tail_expected_observations=N * 0.025,
        scope="Budgets 1 and 4, both algorithms and history conditions; restricted whole evaluation cohort; pointwise intervals only",
    )
    (A / "round2_reselection_bootstrap_validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
