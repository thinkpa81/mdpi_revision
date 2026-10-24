import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "revision/code"))
import original_model_engine as eng

T = ROOT / "revision/tables"
ORIG = ROOT / "original_results/main_result/mdpi-main/01_required_prior_analysis/revision_20260907"
dev = pd.read_csv(ROOT / "private_data/development.csv", dtype={"id": str})
ev = pd.read_csv(ROOT / "private_data/evaluation.csv", dtype={"id": str})
pred = dict(np.load(ROOT / "private_data/refit_all_predictions.npz"))
y = ev.outcome_any_use.to_numpy(int)
pol = pd.read_csv(ROOT / "revision/audit/original_budget_policies.csv")
fixed = pd.read_csv(T / "fixedfit_paired_harm_benefit_inference.csv")
orig_bounds = pd.read_csv(ORIG / "outputs/standardized_policy_bounds.csv")
mde = fixed[
    (fixed.domain == "whole")
    & fixed.family.eq("original_all_budgets_two_references")
    & fixed.model.isin(["ridge::B", "ridge::H_B", "hgb::B", "hgb::H_B"])
    & (
        fixed.history_available & fixed.reference.eq("P0")
        | ~fixed.history_available & fixed.reference.str.endswith("::NONE")
    )
].copy()
mde = mde[
    [
        "algorithm",
        "history_available",
        "domain",
        "family",
        "model",
        "reference",
        "n",
        "delta_log_loss",
        "point_upper95_one_sided",
        "simultaneous_upper95_one_sided",
        "studentized_upper95",
        "approx_MDE80_pointwise",
        "approx_MDE80_simultaneous",
    ]
]
mde["paired_bootstrap_se"] = mde.approx_MDE80_pointwise / (norm.ppf(0.95) + norm.ppf(0.8))
mde["observed_improvement"] = -mde.delta_log_loss
mde["bootstrap_replicates"] = 5000
mde["seed"] = 20261020
mde["scope"] = "conditional on frozen predictions; local normal approximation, not achieved power"
bootstrap_rows = []
for seed in (20260909, 20261020):
    weights = np.random.default_rng(seed).multinomial(
        len(y), np.full(len(y), 1 / len(y)), size=5000
    ).astype(float) / len(y)
    for algo in ("ridge", "hgb"):
        for available in (False, True):
            fam = pol[(pol.algorithm == algo) & (pol.history_available == available)]
            ref = "P0" if available else algo + "::NONE"
            full = algo + "::" + "_".join((["H"] if available else []) + eng.OPTIONAL)
            pairs = list(
                dict.fromkeys(
                    (
                        (r.chosen_model, r0)
                        for _, r in fam.iterrows()
                        for r0 in (full, ref)
                        if r.chosen_model != r0
                    )
                )
            )
            model = algo + "::" + ("H_B" if available else "B")
            j = pairs.index((model, ref))
            keys = list(dict.fromkeys((k for pair in pairs for k in pair)))
            loss = {k: eng.loss_vector(y, pred[k]) for k in keys}
            D = np.column_stack([loss[a] - loss[b] for a, b in pairs])
            delta = D.mean(0)
            with threadpool_limits(1):
                all_boot = weights @ D
            for reps in (2000, 5000):
                boot = all_boot[:reps]
                e = boot - delta
                se = e.std(0, ddof=1)
                cu = np.quantile(np.max(-e, axis=1), 0.95)
                qz = np.quantile(np.max(-e / se, axis=1), 0.95)
                row = dict(
                    algorithm=algo,
                    history_available=available,
                    domain="whole",
                    model=model,
                    reference=ref,
                    family="original_all_budgets_two_references",
                    unique_nonzero_contrasts=len(pairs),
                    bootstrap_replicates=reps,
                    seed=seed,
                    delta_log_loss=delta[j],
                    paired_bootstrap_se=se[j],
                    upper_critical=cu,
                    standardized_upper_critical=qz,
                    unstudentized_upper95=delta[j] + cu,
                    standardized_upper95=delta[j] + qz * se[j],
                    interval="one-sided simultaneous 95% upper bound",
                    method="fixed-bootstrap-SE standardized maximum; not replicate-studentized bootstrap-t",
                )
                if seed == 20260909 and reps == 2000:
                    old = orig_bounds[
                        (orig_bounds.algorithm == algo)
                        & (orig_bounds.history_available == available)
                        & (orig_bounds.domain == "whole")
                        & (orig_bounds.budget == 1)
                    ].iloc[0]
                    row["source_unstudentized_upper95"] = old.primary_upper95_vs_benchmark
                    row["source_standardized_upper95"] = old.standardized_upper95_vs_benchmark
                    assert (
                        abs(row["unstudentized_upper95"] - old.primary_upper95_vs_benchmark)
                        < 1e-12
                    )
                    assert (
                        abs(row["standardized_upper95"] - old.standardized_upper95_vs_benchmark)
                        < 1e-12
                    )
                if seed == 20261020 and reps == 5000:
                    cur = mde[(mde.algorithm == algo) & (mde.history_available == available)].iloc[
                        0
                    ]
                    assert (
                        abs(row["unstudentized_upper95"] - cur.simultaneous_upper95_one_sided)
                        < 1e-12
                    )
                    assert abs(row["standardized_upper95"] - cur.studentized_upper95) < 1e-12
                bootstrap_rows.append(row)
bootstrap = pd.DataFrame(bootstrap_rows)
disjoint = ~ev.id.isin(dev.id).to_numpy()
rounding = []
for domain, mask in [("whole", np.ones(len(ev), bool)), ("person_disjoint", disjoint)]:
    for name, key in [("NONE", "ridge::NONE"), ("P0", "P0")]:
        value = eng.loss_vector(y[mask], pred[key][mask]).mean()
        rounding.append(
            dict(
                domain=domain,
                model=name,
                n=int(mask.sum()),
                raw_log_loss=value,
                format_5_decimals=f"{value:.5f}",
                format_6_decimals=f"{value:.6f}",
                source="raw-data-refit prediction vector, natural-log loss",
                affected_location=(
                    "Supplement Table S8 note: replace 0.69198 with 0.69199"
                    if domain == "person_disjoint" and name == "NONE"
                    else "cross-check"
                ),
            )
        )
rounding = pd.DataFrame(rounding)
sub = pd.read_csv(T / "model_sensitivity_full_model_intervals.csv")
sub = sub[sub.evaluation_group.isin(["low_or_no_hesitation", "no_service_experience"])].copy()
full_dev = pd.read_csv(ROOT / "private_data/full_development.csv", dtype={"id": str})
full_ev = pd.read_csv(ROOT / "private_data/full_evaluation.csv", dtype={"id": str})
fy = full_ev.outcome_any_use.to_numpy(int)
fp0 = eng.p0(full_dev, full_ev)
fn = np.full(len(full_ev), (full_dev.outcome_any_use.sum() + 1) / (len(full_dev) + 2))
for ix, r in sub.iterrows():
    mask = (
        full_ev.hesitation_frequency.isin([3, 4]).to_numpy()
        if r.evaluation_group == "low_or_no_hesitation"
        else full_ev.hesitation_frequency.eq(9).to_numpy()
    )
    a = np.load(
        ROOT / f"private_data/full__{r.algorithm}__{r.representation}.npz", allow_pickle=True
    )
    assert np.array_equal(a["evaluation_id"].astype(str), full_ev.id.to_numpy())
    ll = eng.loss_vector(fy[mask], a["prediction"][mask]).mean()
    bp = fp0 if r.benchmark == "P0" else fn
    bl = eng.loss_vector(fy[mask], bp[mask]).mean()
    assert mask.sum() == r.n and abs(ll - bl - r.delta_log_loss) < 1e-12
    sub.loc[ix, "model_log_loss"] = ll
    sub.loc[ix, "benchmark_log_loss"] = bl
    sub.loc[ix, "events"] = int(fy[mask].sum())
    sub.loc[ix, "prevalence"] = fy[mask].mean()
    sub.loc[ix, "questionnaire_codes"] = (
        "3,4" if r.evaluation_group == "low_or_no_hesitation" else "9"
    )
    sub.loc[ix, "bootstrap_seed"] = (
        2026102121 if r.evaluation_group == "low_or_no_hesitation" else 2026102122
    )
sub["events"] = sub.events.astype(int)
sub["bootstrap_seed"] = sub.bootstrap_seed.astype(int)
sub["superiority_simultaneous95"] = sub.simultaneous_high < 0
sub["family"] = "eight full-model contrasts, separately within each subgroup; fixed predictions"
margins = []
for algo in ("ridge", "hgb"):
    for h in (False, True):
        model = algo + "::" + ("H_B" if h else "B")
        rows = fixed[
            (fixed.domain == "whole")
            & fixed.family.eq("original_all_budgets_two_references")
            & fixed.model.eq(model)
            & fixed.history_available.eq(h)
        ]
        for _, r in rows.iterrows():
            for margin in (0, 0.0025, 0.005, 0.01):
                margins.append(
                    dict(
                        algorithm=algo,
                        history_available=h,
                        model=model,
                        reference=r.reference,
                        delta_log_loss=r.delta_log_loss,
                        simultaneous_upper95_one_sided=r.simultaneous_upper95_one_sided,
                        margin=margin,
                        passes=bool(r.simultaneous_upper95_one_sided < margin),
                        interpretation=(
                            "zero allowance; same strict criterion as superiority"
                            if margin == 0
                            else "positive retrospective non-inferiority allowance"
                        ),
                    )
                )
margins = pd.DataFrame(margins)
frames = {
    "mde_one_item": mde,
    "rare_and_no_experience": sub,
    "rounding": rounding,
    "bootstrap_count_seed": bootstrap,
    "noninferiority_thresholds": margins,
}
for name, df in frames.items():
    df.to_csv(OUT / f"round2_{name}.csv", index=False, float_format="%.17g")
combined = pd.concat([df.assign(table=name) for name, df in frames.items()], ignore_index=True)
combined.to_csv(OUT / "round2_traceability_tables.csv", index=False, float_format="%.17g")
sources = [
    "revision/tables/fixedfit_paired_harm_benefit_inference.csv",
    "revision/tables/model_sensitivity_full_model_intervals.csv",
    "revision/code/refit_and_inference.py",
    "revision/code/model_sensitivity_intervals.py",
    "revision/audit/original_budget_policies.csv",
    "original_results/main_result/mdpi-main/01_required_prior_analysis/revision_20260907/code/run_standardized_sensitivity.py",
]
payload = {
    "scope": "Round 2 traceability and numerical verification; no fitted model or source result modified",
    "tables": {
        name: json.loads(df.to_json(orient="records", double_precision=15))
        for name, df in frames.items()
    },
    "csv_numeric_precision": "17 significant digits; use CSV for full binary64 round-trip precision",
    "mde_formulas": {"pointwise": "(z_0.95 + z_0.80) * SE", "simultaneous": "cU + z_0.80 * SE"},
    "bootstrap_comparison_conclusion": "Both seed and replicate count changed; estimators, family, fixed-SE scaling, resampling unit, and quantile definition did not change.",
    "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
}
(OUT / "round2_traceability_tables.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    "MDE ranges:",
    mde.approx_MDE80_pointwise.min(),
    mde.approx_MDE80_pointwise.max(),
    mde.approx_MDE80_simultaneous.min(),
    mde.approx_MDE80_simultaneous.max(),
)
print(
    bootstrap[(bootstrap.algorithm == "ridge") & ~bootstrap.history_available][
        ["seed", "bootstrap_replicates", "unstudentized_upper95", "standardized_upper95"]
    ].to_string(index=False, float_format=lambda x: f"{x:.12f}")
)
print("Rounding:", rounding.to_dict("records"))
print(
    "Subgroup supported:",
    sub.groupby("evaluation_group").superiority_simultaneous95.sum().to_dict(),
)
print("Saved five component CSVs plus combined CSV/JSON.")
