from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
KEYS = ["algorithm", "history_available", "budget"]


def main():
    validation = json.loads((A / "round2_reselection_bootstrap_validation.json").read_text())
    assert validation["status"] == "passed" and validation["replicates"] == 1000
    q = pd.read_csv(T / "round2_reselection_bootstrap_intervals.csv")
    f = pd.read_csv(T / "round2_reselection_bootstrap_frequencies.csv")
    convergence = pd.read_csv(T / "round2_reselection_monte_carlo_convergence.csv")
    old = pd.read_csv(T / "reselection_bootstrap_intervals.csv")
    results = []
    for _, r in q.iterrows():
        g = f[
            (f.algorithm == r.algorithm)
            & (f.history_available == r.history_available)
            & (f.budget == r.budget)
        ]
        selected = g[g.selected_subset == r.point_subset].iloc[0]
        results.append(
            f"| {('Ridge' if r.algorithm == 'ridge' else 'HGB')} | {('Available' if r.history_available else 'Unavailable')} | {r.budget} | {r.point_subset.replace('_', '+')} | {r.delta_vs_benchmark_point:+.6f} | [{r.delta_vs_benchmark_lower_025:+.6f}, {r.delta_vs_benchmark_upper_975:+.6f}] | [{r.delta_vs_full_lower_025:+.6f}, {r.delta_vs_full_upper_975:+.6f}] | {selected.frequency:.1%} |"
        )
    counts = {}
    for contrast in ["benchmark", "full"]:
        lo = q[f"delta_vs_{contrast}_lower_025"]
        hi = q[f"delta_vs_{contrast}_upper_975"]
        counts[contrast] = int(((lo <= 0) & (hi >= 0)).sum())
    comparison = q.merge(old, on=KEYS, suffixes=("_1000", "_200"))
    shifts = []
    for contrast in ["benchmark", "full"]:
        for side in ["lower_025", "upper_975"]:
            col = f"delta_vs_{contrast}_{side}"
            shifts.extend(abs(comparison[col + "_1000"] - comparison[col + "_200"]).tolist())
    recurrence = []
    for _, r in q.iterrows():
        g = f[
            (f.algorithm == r.algorithm)
            & (f.history_available == r.history_available)
            & (f.budget == r.budget)
            & (f.selected_subset == r.point_subset)
        ]
        recurrence.append(float(g.iloc[0].frequency))
    m = f.monte_carlo_se.max()
    text = f"# Extended participant-union full-process bootstrap: Methods and Results\n\n## Methods\n\nWe extended the focused full-process bootstrap from 200 to **1,000 completed participant-union resamples**. The first 200 resamples and the point estimator were reused after verifying the original fitting-script hash, reconstructed input-data hashes, five-fold specification, hyperparameter grids, seed schedule, and byte-identical checkpoint files. Replicates 201–1,000 used the unchanged fitting function. Reuse avoids recomputation; it does not create additional independent draws from existing results. The final analysis contains 8,000 procedure-specific rows: 1,000 draws for each combination of two algorithms, two history-availability conditions, and two acquisition budgets (1 and 4).\n\nEach replicate draws 2,444 participants with replacement from the union of development and evaluation participants. A participant shared across periods receives the same multinomial multiplicity in both periods; resulting period-specific sample sizes can vary. Five stratified development folds are formed from distinct sampled participants before duplicate copies are expanded, keeping every copy of a participant within the same fold. Each training fold independently estimates imputation, scaling, and categorical encoding. Hyperparameters are retuned, the feasible portfolio is reselected, and the selected model is refitted to the resampled development cohort before evaluation on the corresponding resampled later cohort. No evaluation outcome is used to select or tune a model.\n\nThe search covers all 34 subsets eligible at budgets 1 or 4, including NONE, with P0 additionally eligible in the history-available condition. History-unavailable selection excludes H and P0. Ridge uses C = 0.001, 0.01, 0.1, 1, or 10. HGB uses three or seven leaves, learning rate 0.05, 100 iterations, minimum leaf size 20, L2 regularization 1, and no early stopping. Within a budget, selection minimizes pooled out-of-fold log loss, then actual additional-item count, then lexical portfolio name. The two full-information references, with and without H, are separately retuned and refitted for each algorithm and replicate. Replicate r uses seed 202610200 + r; the unchanged point estimator uses the original fold seed 20260907.\n\nIntervals are the empirical 2.5th and 97.5th percentiles, with linear interpolation. They are pointwise intervals for this full selection-and-fitting procedure, conditional on the fixed two-transition design. They are neither simultaneous intervals nor bootstrap p-values. Positive differences denote higher candidate log loss than the stated reference. Selection recurrence is the proportion of replicates selecting the original point-estimate portfolio; its Monte Carlo standard error is sqrt[p(1−p)/1000]. The accompanying cumulative summaries at 200, 400, 600, 800, and 1,000 replicates, and five disjoint 200-replicate batches, document simulation sensitivity without treating the reused initial 200 as a separate dataset.\n\n## Results\n\nAll 1,000 replicates completed all eight procedures and passed validation. The original 200 replicate rows and the point estimator are preserved exactly. The nominal support in each 2.5% tail increases from approximately five to approximately 25 draws. The maximum observed Monte Carlo standard error of a portfolio selection frequency is {m:.4f} ({100 * m:.2f} percentage points), versus 0.0348 in the initial 200-replicate analysis; the theoretical maximum at 1,000 draws is 0.0158.\n\n| Algorithm | History | Budget | Original portfolio | ΔLL vs benchmark | 95% full-process interval vs benchmark | 95% full-process interval vs full reference | Original-selection recurrence |\n|---|---|---:|---|---:|---|---|---:|\n{chr(10).join(results)}\n\nBenchmark = P0 when history is available and NONE when unavailable. The full-reference interval uses the separately retuned full model in the same history condition. {counts['benchmark']} of eight benchmark-relative intervals and {counts['full']} of eight full-reference intervals include zero. Original-selection recurrence ranges from {min(recurrence):.1%} to {max(recurrence):.1%}. The largest absolute endpoint change between the initial 200 and final 1,000 draws across the 16 contrast intervals is {max(shifts):.6f}; the cumulative and batch tables show the complete progression rather than relying only on this maximum.\n\n## Interpretation and limitations\n\nIncreasing the draw count reduces simulation error and gives each percentile tail more observations. It does not enlarge the participant sample, remove selection instability, establish independent external validity, or extend the scope beyond budgets 1 and 4 in the restricted whole evaluation cohort. Twenty-five expected tail observations still provide finite Monte Carlo precision. Increasing the number of draws does not itself establish nominal frequentist coverage for a discrete portfolio-selection rule. The analysis does not supply full-process simultaneous inference for the entire budget frontier, person-disjoint evaluation, or the full-panel sensitivity population. The unchanged original grids are retained to isolate the effect of increasing the bootstrap draw count; separately expanded model-family sensitivity addresses algorithmic robustness. These intervals describe the adaptive development procedure and must not replace fixed-fit paired harm or benefit tests, which condition on the original fitted models and ask a different question.\n\n## Reproducibility\n\n- Runner: `revision/code/round2_run_selection_bootstrap.py`; unchanged fitting implementation: `revision/code/run_selection_bootstrap.py`; model engine: `revision/code/original_model_engine.py`.\n- Validation: `revision/code/round2_validate_selection_bootstrap.py`; note generation: `revision/code/round2_reselection_report.py`.\n- Aggregate results: `revision/tables/round2_reselection_bootstrap_replicates.csv`, `round2_reselection_bootstrap_intervals.csv`, `round2_reselection_bootstrap_frequencies.csv`, and `round2_reselection_point_estimates.csv`.\n- Simulation diagnostics: `revision/tables/round2_reselection_monte_carlo_convergence.csv` and `round2_reselection_monte_carlo_batches.csv`.\n- Provenance and completion: `revision/audit/round2_reselection_reuse_provenance.json`, `round2_reselection_bootstrap_manifest.json`, and `round2_reselection_bootstrap_validation.json`.\n\nNo individual records, identifiers, or predictions are contained in these aggregate output tables.\n"
    (A / "round2_reselection_bootstrap_results.md").write_text(text)
    (ROOT / "analysis/round2_reselection_methods_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
