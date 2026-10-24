from pathlib import Path
import pandas as pd, numpy as np, json, hashlib

root = Path(__file__).resolve().parents[2]
tables = root / "revision/tables"
r = pd.read_csv(tables / "reselection_bootstrap_replicates.csv")
f = pd.read_csv(tables / "reselection_bootstrap_frequencies.csv")
q = pd.read_csv(tables / "reselection_bootstrap_intervals.csv")
keys = ["algorithm", "history_available", "budget"]
assert len(r) == 1600 and r.replicate.nunique() == 200
assert set(r.replicate) == set(range(1, 201))
assert r.groupby(keys).size().eq(200).all()
assert not r.duplicated(["replicate"] + keys).any()
assert r.selected_items.le(r.budget).all()
assert r[r.history_available == False].selected_subset.str.contains("H|P0", regex=True).sum() == 0
assert np.allclose(r.temporal_log_loss - r.benchmark_log_loss, r.delta_vs_benchmark, atol=1e-12)
assert np.allclose(r.temporal_log_loss - r.full_log_loss, r.delta_vs_full, atol=1e-12)
assert np.allclose(f.groupby(keys).frequency.sum(), 1, atol=1e-12)
assert r.roc_auc.between(0, 1).all()
assert (r.unique_development < r.n_development).all() and (
    r.unique_evaluation < r.n_evaluation
).all()
assert (q.delta_vs_benchmark_lower_025 <= q.delta_vs_benchmark_upper_975).all()
checks = dict(
    status="passed",
    replicates=200,
    procedures=8,
    total_rows=len(r),
    portfolio_frequency_sums_one=True,
    budget_constraints_satisfied=True,
    history_unavailable_excludes_H_and_P0=True,
    reported_differences_recomputed=True,
    all_replicates_have_duplicate_participants=True,
    duplicate_group_disjoint_assertions="Executed inside every development CV split",
    input_development_sha256=hashlib.sha256(
        (root / "private_data/development.csv").read_bytes()
    ).hexdigest(),
    input_evaluation_sha256=hashlib.sha256(
        (root / "private_data/evaluation.csv").read_bytes()
    ).hexdigest(),
    maximum_frequency_monte_carlo_se=float(f.monte_carlo_se.max()),
    bootstrap_file_sha256=hashlib.sha256(
        (tables / "reselection_bootstrap_replicates.csv").read_bytes()
    ).hexdigest(),
)
(root / "revision/audit/reselection_bootstrap_validation.json").write_text(
    json.dumps(checks, indent=2)
)
print(json.dumps(checks, indent=2))
