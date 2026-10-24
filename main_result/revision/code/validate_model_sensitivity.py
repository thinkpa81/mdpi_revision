from pathlib import Path
import sys, json, hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model_sensitivity as ms
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    metrics = pd.read_csv(ms.TABLES / "model_sensitivity_metrics.csv")
    grid = pd.read_csv(ms.TABLES / "model_sensitivity_tuning.csv")
    assert len(metrics) == 192 and len(grid) == 784
    discrepancy = 0
    auc_diff = 0
    models = 0
    for pop in ["restricted", "full"]:
        ev = pd.read_csv(
            ms.PRIVATE / ("evaluation.csv" if pop == "restricted" else "full_evaluation.csv"),
            dtype={"id": str},
        )
        y = ev.outcome_any_use.to_numpy(int)
        for algo in ms.GRIDS:
            for rep in ms.REPS:
                g = grid.query("population==@pop and algorithm==@algo and representation==@rep")
                sel = g.loc[g.selected]
                assert len(sel) == 1 and np.isclose(
                    sel.development_cv_log_loss.iloc[0],
                    g.development_cv_log_loss.min(),
                    atol=1e-14,
                )
                assert len(g) == len(ms.GRIDS[algo])
                arr = np.load(ms.PRIVATE / f"{pop}__{algo}__{rep}.npz", allow_pickle=True)
                p = arr["prediction"]
                assert np.array_equal(arr["evaluation_id"].astype(str), ev.id.to_numpy())
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                row = metrics.query(
                    'population==@pop and algorithm==@algo and representation==@rep and evaluation_group=="all"'
                ).iloc[0]
                discrepancy = max(
                    discrepancy, abs(float(ms.orig.loss_vector(y, p).mean()) - row.log_loss)
                )
                auc_diff = max(
                    auc_diff,
                    abs(ms.auc_ci(y, p)[0] - roc_auc_score(y, p)),
                    abs(row.roc_auc - roc_auc_score(y, p)),
                )
                models += 1
    assert discrepancy < 1e-12 and auc_diff < 1e-12
    hd = pd.read_csv(ms.TABLES / "existing_HD_budget_frontier.csv")
    assert (hd.actual_additional_cost <= hd.budget).all()
    fixed = hd.query('scenario=="fixed_H_D_inclusion"')
    assert fixed.selected_subset.str.split("_").apply(lambda a: "H" in a and "D" in a).all()
    assert len(hd) == 216 and len(fixed) == 108
    inter = pd.read_csv(ms.TABLES / "model_sensitivity_full_model_intervals.csv")
    counts = inter.groupby(["population", "evaluation_group"]).size()
    assert (counts == 8).all() and (inter.replicates == 5000).all()
    assert (inter.simultaneous_low <= inter.delta_log_loss).all() and (
        inter.simultaneous_high >= inter.delta_log_loss
    ).all()
    audit = dict(
        status="passed",
        validated_fitted_models=models,
        model_metrics_rows=len(metrics),
        tuning_configurations=len(grid),
        five_fold_fits=5 * len(grid),
        max_log_loss_reconstruction_difference=discrepancy,
        max_auc_reconstruction_difference=auc_diff,
        existing_HD_scenario_rows=len(hd),
        full_model_bootstrap_family_count=len(counts),
        each_family_contrasts=8,
        bootstrap_replicates=5000,
        probabilities_finite_and_bounded=True,
        minimum_cv_loss_configuration_selected=True,
        evaluation_id_order_matches=True,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (ms.AUDIT / "model_sensitivity_validation.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
