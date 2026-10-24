from pathlib import Path
import pandas as pd, numpy as np

R = Path(__file__).resolve().parents[2]
p = pd.read_csv(R / "revision/audit/original_budget_policies.csv")
i = pd.read_csv(R / "revision/tables/fixedfit_paired_harm_benefit_inference.csv")
i = i[i.family == "original_all_budgets_two_references"]
m = pd.read_csv(R / "revision/tables/refit_all_model_metrics.csv").set_index("model")
y = pd.read_csv(R / "private_data/evaluation.csv").outcome_any_use.to_numpy()
d = pd.read_csv(R / "private_data/development.csv", dtype={"id": str})
e = pd.read_csv(R / "private_data/evaluation.csv", dtype={"id": str})
preds = dict(np.load(R / "private_data/refit_all_predictions.npz"))
rows = []
for _, r in p.iterrows():
    a = r.algorithm
    h = r.history_available
    model = r.chosen_model
    base = "P0" if h else a + "::NONE"
    full = a + "::" + ("H_" if h else "") + "D_B_M_P_C_A_K"
    for domain, mask in [
        ("whole", np.ones(len(e), bool)),
        ("disjoint", ~e.id.isin(set(d.id)).to_numpy()),
    ]:

        def ll(k):
            pp = np.clip(preds[k][mask], 1e-15, 1 - 1e-15)
            yy = y[mask]
            return float(np.mean(-yy * np.log(pp) - (1 - yy) * np.log1p(-pp)))

        row = {
            **r.to_dict(),
            "domain": domain,
            "n": int(mask.sum()),
            "log_loss": ll(model),
            "benchmark": base,
            "benchmark_log_loss": ll(base),
            "full_model": full,
            "full_log_loss": ll(full),
            "full_reference_beats_benchmark": ll(full) < ll(base),
        }
        for ref, name in [(base, "benchmark"), (full, "full")]:
            row["delta_vs_" + name] = ll(model) - ll(ref)
            if model == ref:
                row["lower95_vs_" + name] = row["upper95_vs_" + name] = 0
                row["harm_vs_" + name] = row["benefit_vs_" + name] = False
            else:
                v = i[
                    (i.algorithm == a)
                    & (i.history_available == h)
                    & (i.domain == domain)
                    & (i.model == model)
                    & (i.reference == ref)
                ].iloc[0]
                row["lower95_vs_" + name] = v.simultaneous_lower95_one_sided
                row["upper95_vs_" + name] = v.simultaneous_upper95_one_sided
                row["harm_vs_" + name] = v.harm_detected
                row["benefit_vs_" + name] = v.benefit_detected
        rows.append(row)
s = pd.DataFrame(rows)
s.to_csv(R / "revision/tables/selected_budget_policy_results.csv", index=False)
s[s.budget.isin([0, 1, 4, 30])].to_csv(
    R / "revision/tables/key_budget_policy_results.csv", index=False
)
print(
    s[s.domain == "whole"]
    .groupby(["algorithm", "history_available"])[["harm_vs_benchmark", "benefit_vs_benchmark"]]
    .sum()
)
