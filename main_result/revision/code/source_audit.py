from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "private_data"
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
for d in (P, T, A):
    d.mkdir(parents=True, exist_ok=True)
original = (
    ROOT
    / "original_results/main_result/mdpi-main/01_required_prior_analysis/code/harmonize_data.py"
)
spec = importlib.util.spec_from_file_location("harm", original)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(f, name):
    f.to_csv(T / name, index=False, encoding="utf-8-sig")


def full_transition(b, o, y):
    eligible = b.copy()
    original_hes = eligible.set_index("id").hesitation_frequency
    eligible["hesitation_frequency"] = 1
    x = h.transition(eligible, o, y)
    x["hesitation_frequency"] = x.id.map(original_hes)
    return x


waves = {}
manifest = {
    "source_waves": [],
    "source_files": {2023: "2023_data.xlsx", 2024: "2024_data.xlsx", 2025: "2025_data.xlsx"},
    "source_script_sha256": digest(original),
}
for y in (2023, 2024, 2025):
    source = ROOT / "private_raw" / f"{y}_data.xlsx"
    raw = h.read_source(source)
    idc = h.SPECS[y]["id"]
    assert raw[idc].notna().all() and raw[idc].is_unique
    waves[y], quality = h.harmonize(raw, y)
    waves[y].to_csv(P / f"wave_{y}.csv", index=False)
    manifest["source_waves"].append(
        {
            "year": y,
            "rows": len(raw),
            "columns": len(raw.columns),
            "file_bytes": source.stat().st_size,
            "sha256": digest(source),
            "quality": quality,
        }
    )
    print("read", y, len(raw), len(raw.columns), flush=True)
frames = {
    "development": h.transition(waves[2023], waves[2024], 2023),
    "evaluation": h.transition(waves[2024], waves[2025], 2024),
    "full_development": full_transition(waves[2023], waves[2024], 2023),
    "full_evaluation": full_transition(waves[2024], waves[2025], 2024),
}
for prefix in ("", "full_"):
    d, e = (frames[prefix + "development"], frames[prefix + "evaluation"])
    e["seen_development_id"] = e.id.isin(set(d.id)).astype(int)
manifest["analysis_samples"] = {}
for k, f in frames.items():
    f.to_csv(P / (k + ".csv"), index=False)
    manifest["analysis_samples"][k] = {
        "n": len(f),
        "events": int(f.outcome_any_use.sum()),
        "event_rate": float(f.outcome_any_use.mean()),
        "monthly_events": int(f.outcome_monthly_use.sum()),
        "valid_longitudinal_weight_n": int(
            (f.outcome_long_weight.notna() & (f.outcome_long_weight > 0)).sum()
        ),
        "missing_predictor_cells": int(
            f[[v for g in h.GROUPS.values() for v in g["numeric"] + g["categorical"]]]
            .isna()
            .sum()
            .sum()
        ),
        "sha256": digest(P / (k + ".csv")),
    }
(A / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("CSV_READY", json.dumps(manifest["analysis_samples"]), flush=True)
write(
    pd.DataFrame([{"sample": k, **v} for k, v in manifest["analysis_samples"].items()]),
    "sample_reconstruction.csv",
)
dev, eva = (frames["development"], frames["evaluation"])
dcols = {
    "gender": range(1, 3),
    "age_group": range(1, 8),
    "education": range(1, 9),
    "region": range(1, 18),
}
all_samples = {
    **frames,
    "evaluation_overlap": eva[eva.seen_development_id == 1],
    "evaluation_nonoverlap": eva[eva.seen_development_id == 0],
}
rows = []
for sn, f in all_samples.items():
    for col, codes in dcols.items():
        for code in list(codes) + ["Missing"]:
            n = int(f[col].isna().sum() if code == "Missing" else (f[col] == code).sum())
            rows.append(
                {
                    "sample": sn,
                    "variable": col,
                    "code": code,
                    "n": n,
                    "denominator": len(f),
                    "proportion": n / len(f),
                }
            )
write(pd.DataFrame(rows), "demographic_category_distributions.csv")
pairedD = dev[["id"] + list(dcols)].merge(
    eva[["id"] + list(dcols)], on="id", suffixes=("_2023", "_2024"), validate="one_to_one"
)
rows = []
trans = []
for col in dcols:
    x, y = (pairedD[col + "_2023"], pairedD[col + "_2024"])
    valid = x.notna() & y.notna()
    rows.append(
        {
            "variable": col,
            "paired_n": len(pairedD),
            "valid_pairs": int(valid.sum()),
            "changed_n": int(((x != y) & valid).sum()),
            "changed_proportion": float(((x != y) & valid).sum() / valid.sum()),
        }
    )
    for (a, b), n in pairedD.groupby([col + "_2023", col + "_2024"], dropna=False).size().items():
        trans.append({"variable": col, "baseline_category": a, "next_category": b, "n": n})
write(pd.DataFrame(rows), "demographic_within_person_changes.csv")
write(pd.DataFrame(trans), "demographic_within_person_transitions.csv")
rows = []
for sn, f in all_samples.items():
    for col in [
        "prior_any_use",
        "outcome_any_use",
        "prior_monthly_use",
        "outcome_monthly_use",
        "benefit_expectation",
        "hesitation_frequency",
    ]:
        rows.append(
            {
                "sample": sn,
                "variable": col,
                "n": len(f),
                "nonmissing": int(f[col].notna().sum()),
                "mean": float(f[col].mean()),
                "sd": float(f[col].std()),
            }
        )
write(pd.DataFrame(rows), "overlap_and_full_panel_comparison.csv")
w23 = waves[2023].set_index("id")
w24 = waves[2024].set_index("id")
entry = []
for idv in eva.id:
    if idv in set(dev.id):
        reason = "Present in restricted development"
    elif idv not in w23.index:
        reason = "Not observed in 2023"
    elif not pd.notna(w23.loc[idv, "hesitation_frequency"]):
        reason = "2023 hesitation missing"
    elif w23.loc[idv, "hesitation_frequency"] not in (1, 2):
        reason = "Observed in 2023 but outside hesitation eligibility"
    elif idv not in w24.index:
        reason = "Not linked to 2024"
    elif not pd.notna(w24.loc[idv, "any_use"]):
        reason = "2024 outcome missing"
    else:
        raise AssertionError("unexplained nonoverlap")
    entry.append(reason)
write(
    pd.Series(entry)
    .value_counts()
    .rename_axis("entry_reason")
    .rename("n")
    .reset_index()
    .assign(evaluation_n=len(eva)),
    "evaluation_entry_decomposition.csv",
)
rows = []
for sn, f in frames.items():
    for code, n in f.outcome_frequency.value_counts(dropna=False).sort_index().items():
        rows.append({"sample": sn, "frequency_code": code, "n": int(n), "proportion": n / len(f)})
write(pd.DataFrame(rows), "outcome_frequency_distribution_revision.csv")
pairs = dev[
    ["id", "outcome_any_use", "outcome_monthly_use", "outcome_regular_use", "outcome_frequency"]
].merge(
    eva[
        [
            "id",
            "outcome_any_use",
            "outcome_monthly_use",
            "outcome_regular_use",
            "outcome_frequency",
        ]
    ],
    on="id",
    suffixes=("_2024", "_2025"),
    validate="one_to_one",
)
rows = []
trans = []
for col in ["outcome_any_use", "outcome_monthly_use", "outcome_regular_use"]:
    a, b = (pairs[col + "_2024"], pairs[col + "_2025"])
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    rows.append(
        {
            "outcome": col,
            "paired_n": len(pairs),
            "prevalence_2024": float(a.mean()),
            "prevalence_2025": float(b.mean()),
            "change_2025_minus_2024": float((b - a).mean()),
            "no_to_yes": n01,
            "yes_to_no": n10,
            "discordant_n": n01 + n10,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1
            ),
            "interpretation": "Paired response change; time and wording effects cannot be separated",
        }
    )
for (a, b), n in (
    pairs.groupby(["outcome_frequency_2024", "outcome_frequency_2025"]).size().items()
):
    trans.append({"frequency_2024": a, "frequency_2025": b, "n": int(n)})
write(pd.DataFrame(rows), "paired_outcome_change_diagnostics.csv")
write(pd.DataFrame(trans), "paired_outcome_frequency_transitions.csv")
checks = []
for by, oy, label in [(2023, 2024, "development"), (2024, 2025, "evaluation")]:
    b, o = (waves[by], waves[oy])
    ids = set(b.loc[b.hesitation_frequency.isin([1, 2]), "id"]) & set(
        o.loc[o.any_use.notna(), "id"]
    )
    checks.append(
        {
            "check": label + "_independent_id_set",
            "pass": ids == set(frames[label].id),
            "n": len(ids),
        }
    )
    assert ids == set(frames[label].id)
checks.append(
    {
        "check": "restricted_overlap_n",
        "pass": len(set(dev.id) & set(eva.id)) == 641,
        "n": len(set(dev.id) & set(eva.id)),
    }
)
for sn in frames:
    checks.append(
        {
            "check": "no_age7_in_" + sn,
            "pass": bool(frames[sn].age_group.eq(7).sum() == 0),
            "n": int(frames[sn].age_group.eq(7).sum()),
        }
    )
(A / "source_validation.json").write_text(
    json.dumps(
        {
            "checks": checks,
            "scope": "The raw authorized XLSX Numeric worksheets were read directly. Harmonization was independently checked by ID sets. No row-level records are included in the public results.",
            "measurement_limit": "Single self-reported outcome and changed wording are confounded with wave; these diagnostics do not establish measurement invariance.",
        },
        indent=2,
    )
)
print("DONE", flush=True)
