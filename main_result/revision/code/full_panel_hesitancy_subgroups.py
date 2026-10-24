from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model_sensitivity as ms
import numpy as np, pandas as pd


def main():
    dev = pd.read_csv(ms.PRIVATE / "full_development.csv", dtype={"id": str})
    ev = pd.read_csv(ms.PRIVATE / "full_evaluation.csv", dtype={"id": str})
    y = ev.outcome_any_use.to_numpy(int)
    groups = {
        "low_or_no_hesitation": ev.hesitation_frequency.isin([3, 4]).to_numpy(),
        "no_service_experience": (ev.hesitation_frequency == 9).to_numpy(),
    }
    p0 = ms.orig.p0(dev, ev)
    pn = np.full(len(ev), (dev.outcome_any_use.sum() + 1) / (len(dev) + 2))
    rows = []
    for algo in ms.GRIDS:
        for rep in ms.REPS:
            arr = np.load(ms.PRIVATE / f"full__{algo}__{rep}.npz", allow_pickle=True)
            p = arr["prediction"]
            assert np.array_equal(arr["evaluation_id"].astype(str), ev.id.to_numpy())
            for group, sel in groups.items():
                row = dict(
                    population="full",
                    algorithm=algo,
                    representation=rep,
                    evaluation_group=group,
                    **ms.met(y[sel], p[sel]),
                )
                for b, bp in [("P0", p0), ("NONE", pn)]:
                    row[f"delta_vs_{b}"] = float(
                        (
                            ms.orig.loss_vector(y[sel], p[sel])
                            - ms.orig.loss_vector(y[sel], bp[sel])
                        ).mean()
                    )
                rows.append(row)
    for b, bp in [("P0", p0), ("NONE", pn)]:
        for group, sel in groups.items():
            rows.append(
                dict(
                    population="full",
                    algorithm="benchmark",
                    representation=b,
                    evaluation_group=group,
                    **ms.met(y[sel], bp[sel]),
                )
            )
    pd.DataFrame(rows).to_csv(ms.TABLES / "full_panel_hesitancy_subgroup_metrics.csv", index=False)


if __name__ == "__main__":
    main()
