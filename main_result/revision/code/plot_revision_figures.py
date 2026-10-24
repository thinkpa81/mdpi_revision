from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "revision/tables"
OUT = ROOT / "JTAER_Figures"
BLUE = "#23618C"
ORANGE = "#C06332"
GREEN = "#31836C"
GRAY = "#5E6871"
PURPLE = "#7C60A3"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9DA5AC",
        "axes.linewidth": 0.6,
        "grid.color": "#E0E5E8",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
MANIFEST = []
CAPTIONS = []


def read(name):
    return pd.read_csv(T / name)


def save(fig, num, title, caption, data):
    stem = f"Figure_{num:02d}"
    fig.savefig(OUT / (stem + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / (stem + ".svg"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / "preview" / (stem + ".png"), dpi=120, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    data.to_csv(OUT / "data" / (stem + ".csv"), index=False)
    MANIFEST.append(
        dict(
            figure=stem,
            title=title,
            rows=len(data),
            data_sha256=hashlib.sha256((OUT / "data" / (stem + ".csv")).read_bytes()).hexdigest(),
        )
    )
    CAPTIONS.append(f"## {stem} {title}\n\n{caption}\n")


def forest(ax, vals, lo, hi, labels, color=BLUE, xlabel="Difference in log loss"):
    yy = np.arange(len(vals))
    v = np.asarray(vals)
    l = np.asarray(lo)
    u = np.asarray(hi)
    assert np.isfinite(np.r_[v, l, u]).all() and np.all(l <= v) and np.all(v <= u)
    ax.errorbar(
        v,
        yy,
        xerr=np.vstack([v - l, u - v]),
        fmt="o",
        color=color,
        capsize=3,
        markersize=5,
        lw=1.4,
    )
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set_yticks(yy, labels)
    ax.set_ylim(len(vals) - 0.5, -0.5)
    ax.grid(axis="x")
    ax.set_xlabel(xlabel)


def original_reference():
    d = read("core_models_metrics_with_auc_ci.csv")
    d = d[d.domain.eq("whole")].drop_duplicates("model")
    keys = [
        "P0",
        "ridge::NONE",
        "ridge::H_D_B_M_P_C_A_K",
        "hgb::H_D_B_M_P_C_A_K",
        "ridge::D_B_M_P_C_A_K",
        "hgb::D_B_M_P_C_A_K",
    ]
    q = d.set_index("model").loc[keys].reset_index()
    labels = [
        "Prior-use benchmark P0",
        "Prevalence benchmark NONE",
        "Ridge full, with H",
        "HGB full, with H",
        "Ridge full, without H",
        "HGB full, without H",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    yy = np.arange(6)
    ax.errorbar(
        q.log_loss,
        yy,
        xerr=np.vstack([q.log_loss - q.log_loss_lower95, q.log_loss_upper95 - q.log_loss]),
        fmt="o",
        color=BLUE,
        capsize=3,
    )
    ax.axvline(q.log_loss.iloc[0], color=GREEN, ls="--", lw=1, label="P0")
    ax.axvline(q.log_loss.iloc[1], color=GRAY, ls=":", lw=1, label="NONE")
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Temporal log loss (lower is better)")
    ax.grid(axis="x")
    ax.legend(loc="upper right")
    save(
        fig,
        1,
        "Full models and no-additional-collection benchmarks",
        "Original frozen models were refitted from the 2023–2025 source workbooks. Temporal evaluation includes 1,520 respondents. Bars are pointwise 95% participant-bootstrap intervals conditional on fitted predictions (2,000 repetitions). These absolute-loss intervals do not replace paired model-comparison tests. H denotes prior self-reported use.",
        q,
    )


def budget_harm():
    k = read("key_budget_policy_results.csv")
    k = k[(k.domain == "whole") & k.budget.isin([1, 4])]
    inf = read("fixedfit_paired_harm_benefit_inference.csv")
    inf = inf[(inf.domain == "whole") & (inf.family == "original_all_budgets_two_references")]
    rows = []
    for _, r in k.iterrows():
        z = inf[
            (inf.algorithm == r.algorithm)
            & (inf.history_available == r.history_available)
            & (inf.model == r.source_model)
            & (inf.reference == r.benchmark)
        ].iloc[0]
        rows.append(
            dict(
                algorithm=r.algorithm,
                history_available=r.history_available,
                budget=r.budget,
                subset=r.selected_subset,
                delta=r.delta_vs_benchmark,
                lower=z.simultaneous_lower95_two_sided,
                upper=z.simultaneous_upper95_two_sided,
            )
        )
    q = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharex=True)
    for ax, h in zip(axs, [True, False]):
        s = q[q.history_available == h]
        labels = [
            f"{r.algorithm.upper()} · budget {r.budget} ({r.subset.replace('_', ' + ')})"
            for r in s.itertuples()
        ]
        forest(ax, s.delta, s.lower, s.upper, labels)
        ax.set_title("Prior use available" if h else "Prior use unavailable")
    save(
        fig,
        2,
        "Low-budget selections can harm temporal prediction",
        "Delta equals selected-model loss minus P0 (history available) or NONE (history unavailable); positive values indicate harm. Bars are two-sided simultaneous 95% max-absolute centered-bootstrap intervals within each original algorithm × information-availability comparison family, spanning all selected budgets and both submitted references. There were 5,000 evaluation-participant bootstrap repetitions. The eight displayed contrasts are not a newly restricted comparison family.",
        q,
    )


def contribution():
    q = read("attribution_per_item.csv")
    q = q[q.domain == "whole"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.3), layout="constrained", sharex=True)
    labels = {
        "D": "Demographics (4)",
        "B": "Expected benefit (1)",
        "M": "Data management (1)",
        "P": "Safeguards (11)",
        "C": "Consent response (1)",
        "A": "Generative AI use (1)",
        "K": "Digital skills (11)",
    }
    for i, alg in enumerate(["ridge", "hgb"]):
        for j, h in enumerate([True, False]):
            ax = axs[i, j]
            s = q[(q.algorithm == alg) & (q.history_fixed == h)]
            forest(
                ax,
                s.loss_reduction_per_item,
                s.per_item_simultaneous95_lower,
                s.per_item_simultaneous95_upper,
                [labels[b] for b in s.block],
                GREEN,
                "Log-loss reduction per questionnaire item",
            )
            ax.set_title(alg.upper() + (" · H present" if h else " · H absent"))
    save(
        fig,
        3,
        "Predictive contribution per questionnaire item",
        "Exact Shapley allocation of held-out log-loss reduction over seven information blocks, divided by original item count (in parentheses). Positive values indicate improved prediction. H is fixed present or absent. Bars are simultaneous 95% intervals from 5,000 fixed-prediction bootstrap repetitions, jointly covering 56 block × algorithm × history × evaluation-domain contrasts. Only the whole-sample domain is displayed. This division is an efficiency diagnostic, not a cost-weighted Shapley game, causal attribution or individual-feature SHAP.",
        q,
    )


def calibration():
    d = read("calibration_decomposition.csv")
    keys = [
        f"{a}::{s}"
        for a in ["ridge", "hgb"]
        for s in ["H_D", "D", "H_D_B_M_P_C_A_K", "D_B_M_P_C_A_K"]
    ]
    q = d[d["sample"].eq("whole")].set_index("source_model").loc[keys].reset_index()
    labels = [
        f"{a.upper()} · {s}"
        for a in ["ridge", "hgb"]
        for s in ["H + D", "D", "Full with H", "Full without H"]
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), layout="constrained", sharey=True)
    for ax, col, title in zip(
        axs,
        ["affine_miscalibration_component", "affine_resolution_component"],
        ["Apparent recalibration improvement", "Resolution after affine recalibration"],
    ):
        forest(
            ax,
            q[col],
            q[col + "_ci_low"],
            q[col + "_ci_high"],
            labels,
            BLUE if col.endswith("component") else ORANGE,
            "Log-loss component",
        )
        ax.set_title(title)
    save(
        fig,
        4,
        "Calibration and ranking diagnostics",
        "Raw log loss = evaluation outcome entropy − affine logistic resolution + affine logistic miscalibration. These are components relative to apparent intercept-and-slope logistic recalibration, not a nonparametric decomposition. Bars are pointwise 95% percentile intervals from 500 participant bootstrap samples, each refitting the recalibrator. All diagnostics use evaluation outcomes. Cross-fitted recalibration and AUC appear in the accompanying source table; they should guide interpretation of optimistic apparent gains.",
        q,
    )


def holdout():
    d = read("within_period_holdout_summary.csv")
    q = d[d.policy.eq("budget_4")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharex=True, sharey=True)
    for ax, h in zip(axs, ["history_available", "history_unavailable"]):
        for offset, sample, col, label in [
            (-0.13, "within_period_holdout", BLUE, "Within-period holdout"),
            (0.13, "matched_training_temporal", ORANGE, "Later-period evaluation"),
        ]:
            s = (
                q[(q.availability == h) & (q["sample"] == sample)]
                .set_index("algorithm")
                .loc[["ridge", "hgb"]]
            )
            m = s.delta_log_loss_vs_benchmark_mean
            l = s.delta_log_loss_vs_benchmark_split_q025
            u = s.delta_log_loss_vs_benchmark_split_q975
            ax.errorbar(
                m,
                np.arange(2) + offset,
                xerr=np.vstack([m - l, u - m]),
                fmt="o",
                color=col,
                capsize=3,
                label=label,
            )
        ax.axvline(0, ls="--", color=GRAY, lw=1)
        ax.set_yticks([0, 1], ["Ridge", "HGB"])
        ax.set_ylim(1.5, -0.5)
        ax.grid(axis="x")
        ax.set_title(
            "Prior use available" if h == "history_available" else "Prior use unavailable"
        )
        ax.set_xlabel("Budget-4 loss minus benchmark loss")
    axs[1].legend(loc="upper left", bbox_to_anchor=(0, -0.2))
    save(
        fig,
        5,
        "Within-period and temporal validation at matched training size",
        "Twenty stratified 70/30 development splits; each uses inner five-fold tuning and budget-4 subset selection. The same trained model is evaluated on the 470-person within-period holdout and on the later 1,520-person cohort. Points are mean paired loss differences; bars span the 2.5th–97.5th percentiles across splits. These bars represent split variability, not confidence intervals. Differences across evaluation periods combine wording, time and sample-composition effects.",
        q,
    )


def bootstrap():
    path = T / "reselection_bootstrap_intervals.csv"
    if not path.exists():
        return
    q = read(path.name).sort_values(
        ["history_available", "algorithm", "budget"], ascending=[False, False, True]
    )
    freq = read("reselection_bootstrap_frequencies.csv")
    labels = [
        f"{r.algorithm.upper()} · {('H available' if r.history_available else 'H unavailable')} · budget {r.budget}"
        for r in q.itertuples()
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]}
    )
    rates = []
    for r in q.itertuples():
        f = freq[
            (freq.algorithm == r.algorithm)
            & (freq.history_available == r.history_available)
            & (freq.budget == r.budget)
            & (freq.selected_subset == r.point_subset)
        ]
        rates.append(float(f.frequency.iloc[0]) if len(f) else 0)
    yy = np.arange(len(q))
    axs[0].barh(yy, rates, color=BLUE, height=0.6)
    axs[0].set_yticks(yy, labels)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 1.15)
    axs[0].set_xlabel("Frequency of original selected portfolio")
    axs[0].grid(axis="x")
    for y, v in zip(yy, rates):
        axs[0].text(v + 0.02, y, f"{v:.1%}", va="center", fontsize=9)
    vals = q.delta_vs_benchmark_point.to_numpy()
    lo = q.delta_vs_benchmark_lower_025.to_numpy()
    hi = q.delta_vs_benchmark_upper_975.to_numpy()
    axs[1].hlines(yy, lo, hi, color=ORANGE, lw=1.6)
    axs[1].plot(vals, yy, "o", color=ORANGE)
    axs[1].axvline(0, ls="--", color=GRAY, lw=1)
    axs[1].set_yticks(yy, [""] * len(q))
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Selected-model loss minus benchmark loss")
    axs[1].grid(axis="x")
    q = q.copy()
    q["original_portfolio_selection_frequency"] = rates
    reps = int(q.replicates.min())
    save(
        fig,
        6,
        "Uncertainty after repeating selection and model fitting",
        f"{reps} participant-union bootstrap repetitions. Shared participants receive common multiplicities across periods; copies remain in the same inner fold. All 34 portfolios eligible at budgets 1 and 4, P0, preprocessing and original-grid tuning are reconsidered in each repetition. Left: frequency of the original selected portfolio. Right: original point estimate and pointwise 95% percentile interval for the complete procedure. These intervals are exploratory and are not multiplicity-adjusted. The broader budget frontier has fixed-prediction inference only.",
        q,
    )


def expanded():
    path = T / "model_sensitivity_full_model_intervals.csv"
    if not path.exists():
        return
    d = read(path.name)
    q = d[d.evaluation_group.eq("all")]
    order = pd.MultiIndex.from_product(
        [["ridge", "hgb", "rf", "extratrees"], ["full_H", "full_noH"]],
        names=["algorithm", "representation"],
    )
    labels = [
        a + " · " + h
        for a in ["Ridge", "HGB", "Random forest", "Extra trees"]
        for h in ["with H", "without H"]
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(11, 5.4), layout="constrained", sharex=True, sharey=True
    )
    for ax, pop, title, col in zip(
        axs, ["restricted", "full"], ["Privacy-hesitant panel", "Full linked panel"], [BLUE, GREEN]
    ):
        s = q[q.population.eq(pop)].set_index(["algorithm", "representation"]).loc[order]
        forest(
            ax,
            s.delta_log_loss,
            s.simultaneous_low,
            s.simultaneous_high,
            labels,
            col,
            "Full-model loss minus benchmark loss",
        )
        ax.set_title(title)
    save(
        fig,
        7,
        "Additional information value varies by model and population",
        "Four algorithms were tuned with development-only five-fold validation using wider ridge/HGB grids and random-forest/extra-trees grids. Each panel includes eight contrasts: full with H versus P0 and full without H versus NONE across four algorithms. Bars are simultaneous two-sided 95% max-absolute centered paired-bootstrap intervals within each panel (5,000 repetitions), conditional on fitted predictions. Development/evaluation sample sizes are 1,565/1,520 (restricted) and 3,895/3,741 (full linked panel). These post-review comparisons reuse the evaluation period and do not establish population-wide collection effects.",
        q,
    )
    hetero = T / "model_sensitivity_hesitancy_heterogeneity.csv"
    if hetero.exists():
        z = read(hetero.name)
        z = (
            z[z.in_full_model_family]
            .set_index(["algorithm", "representation"])
            .loc[order]
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8.6, 5), layout="constrained")
        forest(
            ax,
            z.difference,
            z.simultaneous_low,
            z.simultaneous_high,
            labels,
            PURPLE,
            "Loss difference in hesitant group minus loss difference in other respondents",
        )
        save(
            fig,
            8,
            "Predictive-value differences between evaluation groups",
            "The same full-panel-trained models are evaluated in baseline privacy-hesitant respondents (n=1,520; hesitation codes 1–2) and other linked respondents (n=2,221; codes 3, 4 and 9). The horizontal contrast compares full-minus-benchmark log-loss differences between these groups. Positive values indicate less benefit or greater harm in the hesitant group. Bars are simultaneous two-sided 95% bootstrap intervals across eight full-model contrasts (5,000 repetitions within each group). This is conditional predictive-performance heterogeneity, not a causal moderator effect.",
            z,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "preview").mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    original_reference()
    budget_harm()
    contribution()
    calibration()
    holdout()
    bootstrap()
    expanded()
    (OUT / "FIGURE_CAPTIONS.md").write_text(
        "# Reanalysis figure captions\n\n" + "\n".join(CAPTIONS)
    )
    (OUT / "figure_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    if args.final:
        assert len(MANIFEST) == 8, "Final bootstrap and expanded-model figures must exist."
    print(json.dumps(MANIFEST, indent=2))


if __name__ == "__main__":
    main()
