from pathlib import Path
import json
import numpy as np
import pandas as pd
import round2_figures as f
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
T = R / "revision/tables"
q = pd.read_csv(T / "round2_reselection_bootstrap_intervals.csv").sort_values(
    ["history_available", "algorithm", "budget"], ascending=[False, False, True]
)
assert len(q) == 8 and (q.replicates == 1000).all()
manifest = json.loads(
    (R / "revision/audit/round2_reselection_bootstrap_manifest.json").read_text()
)
assert manifest["status"] == "complete" and manifest["completed_replicates"] == 1000
freq = pd.read_csv(T / "round2_reselection_bootstrap_frequencies.csv")
labels = [
    f"{r.algorithm.upper()} · {('H available' if r.history_available else 'H unavailable')} · budget {r.budget}"
    for r in q.itertuples()
]
rates = []
for r in q.itertuples():
    row = freq[
        (freq.algorithm == r.algorithm)
        & (freq.history_available == r.history_available)
        & (freq.budget == r.budget)
        & (freq.selected_subset == r.point_subset)
    ]
    assert len(row) == 1
    rates.append(float(row.frequency.iloc[0]))
plt.rcParams.update(
    {"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12}
)
fig, ax = plt.subplots(
    1, 2, figsize=(12, 5.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]}
)
y = np.arange(8)
ax[0].barh(y, rates, color=f.BLUE, height=0.6)
ax[0].set_yticks(y, labels)
ax[0].invert_yaxis()
ax[0].set_xlim(0, 1.15)
ax[0].set_xlabel("Frequency of original selected portfolio")
ax[0].grid(axis="x")
for yy, v in zip(y, rates):
    ax[0].text(v + 0.02, yy, f"{v:.1%}", va="center", fontsize=12)
ax[1].hlines(
    y, q.delta_vs_benchmark_lower_025, q.delta_vs_benchmark_upper_975, color=f.ORANGE, lw=1.6
)
ax[1].plot(q.delta_vs_benchmark_point, y, "o", color=f.ORANGE)
ax[1].axvline(0, ls="--", color=f.GRAY, lw=1)
ax[1].set_yticks(y, [""] * 8)
ax[1].invert_yaxis()
ax[1].set_xlabel("Selected-model loss minus benchmark loss")
ax[1].grid(axis="x")
ax[1].set_ylim(ax[0].get_ylim())
q = q.copy()
q["original_portfolio_selection_frequency"] = rates
caption = "One thousand participant-union repetitions rerun preprocessing, five-fold tuning, selection among all 34 focal-budget candidates plus P0, and final fitting; full references are also retuned. Shared participants receive common multiplicities across periods, with copies grouped within inner folds. Left: recurrence of the original portfolio. Right: original point estimate and pointwise percentile 95% interval of the complete procedure. The analysis covers budgets 1 and 4 in the restricted whole temporal cohort; intervals are not adjusted for multiple comparisons. About 25 resamples lie in each 2.5% tail."
for d in [f.OUT, f.OUT / "data", f.OUT / "preview"]:
    d.mkdir(parents=True, exist_ok=True)
f.save(
    fig,
    "Round2_Figure_04_Selection_Uncertainty",
    "Uncertainty after repeating model development",
    caption,
    q,
    [
        T / "round2_reselection_bootstrap_intervals.csv",
        T / "round2_reselection_bootstrap_frequencies.csv",
    ],
)
(R / "revision/audit/round2_bootstrap_figure_validation.json").write_text(
    json.dumps(f.CHECKS, indent=2)
)
(f.OUT / "Round2_Bootstrap_Caption.md").write_text("\n".join(f.CAPTIONS))
print("Completed validated 1,000-repeat figure")
