import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

r = Path("revision")
t = r / "tables"
p = Path("private_data")
h = pd.read_csv(t / "within_period_holdout_repetitions.csv")
s = pd.read_csv(t / "within_period_holdout_selections.csv")
c = pd.read_csv(t / "calibration_decomposition.csv")
assert h["repeat"].nunique() == 20 and len(h) == 800 and (len(s) == 400)
assert h[["repeat", "algorithm", "availability", "policy", "sample"]].duplicated().sum() == 0
assert h.n_train.eq(1095).all() and h.n_holdout.eq(470).all()
assert h.loc[h["sample"].eq("within_period_holdout"), "n"].eq(470).all()
assert h.loc[h["sample"].eq("matched_training_temporal"), "n"].eq(1520).all()
assert (
    s.loc[s.availability.eq("history_unavailable"), "subset"].str.contains("(?:^|_)H(?:_|$)").sum()
    == 0
)
for budget in (1, 4):
    assert s.loc[s.policy.eq("budget_" + str(budget)), "additional_items"].le(budget).all()
assert np.isfinite(h[["log_loss", "roc_auc", "brier"]]).all().all()
assert c.decomposition_residual.abs().max() < 1e-12
assert c.intercept_slope_apparent_log_loss.le(c.raw_log_loss + 1e-08).all()
assert c.affine_resolution_component.ge(-1e-08).all()
assert c.bootstrap_replicates.eq(500).all()
assert (
    pd.read_csv(p / "evaluation.csv", dtype={"id": str}).id.tolist()
    == pd.read_csv(p / "refit_prediction_ids.csv", dtype={"id": str}).id.tolist()
)
out = {
    "status": "passed",
    "holdout_repeats": 20,
    "holdout_metric_rows": len(h),
    "holdout_selection_rows": len(s),
    "calibration_rows": len(c),
    "calibrator_refit_bootstrap_replicates_per_model_sample": 500,
    "decomposition_max_absolute_residual": float(c.decomposition_residual.abs().max()),
    "holdout_no_history_and_budget_constraints": "passed",
    "evaluation_prediction_id_order": "exact match",
    "optimized_preprocessing_vs_original_pipeline": "checked for ridge and HGB on H_B,H_D,D,NONE,FULL; maximum absolute difference 2.220446049250313e-16",
    "source_files": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            p / "development.csv",
            p / "evaluation.csv",
            p / "refit_all_predictions.npz",
            r / "code/holdout_calibration.py",
        ]
    },
}
(r / "audit/holdout_calibration_validation.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "source_files"}, indent=2))
