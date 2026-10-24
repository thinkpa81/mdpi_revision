from pathlib import Path
import json, pandas as pd, numpy as np

root = Path(__file__).resolve().parents[2]
new = json.loads((root / "private_data/reselection_checkpoints/replicate_0000.json").read_text())[
    "rows"
]
old = pd.read_csv(
    root
    / "original_results/main_result/mdpi-main/01_required_prior_analysis/revision_20260907/outputs/temporal_budget_policy_metrics.csv"
)
audit = []
for r in new:
    avail = "history_available" if r["history_available"] else "history_unavailable"
    o = old[
        (old.algorithm == r["algorithm"])
        & (old.availability == avail)
        & (old.budget == r["budget"])
        & (old.evaluation_domain == "all")
    ].iloc[0]
    assert o.selected_subset == r["selected_subset"]
    for current, original in [
        ("temporal_log_loss", "log_loss"),
        ("delta_vs_benchmark", "delta_baseline_log_loss"),
        ("delta_vs_full", "delta_full_log_loss"),
        ("cv_log_loss", "development_oof_log_loss"),
    ]:
        diff = float(r[current] - o[original])
        audit.append(
            dict(
                algorithm=r["algorithm"],
                history_available=r["history_available"],
                budget=r["budget"],
                metric=current,
                recomputed=r[current],
                original=o[original],
                absolute_difference=abs(diff),
            )
        )
        assert abs(diff) < 1e-10
pd.DataFrame(audit).to_csv(
    root / "revision/audit/reselection_point_original_reproduction.csv", index=False
)
print("All 32 numerical comparisons agree within 1e-10; all eight selections agree.")
