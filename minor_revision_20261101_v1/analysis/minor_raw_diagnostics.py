from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import logit
from scipy.stats import norm


SPECS = {
    2023: {"id": "ID", "use": "q1_8", "hesitation": "q15_6", "A": "q23"},
    2024: {"id": "id", "use": "Q1_8", "hesitation": "Q15_6", "A": "Q24"},
    2025: {"id": "id", "use": "q1_8", "hesitation": "q14_6", "A": "q20"},
}
EXPECTED_SOURCE_HASHES = {
    2023: "c38782a595dc9ee898cb64a3ebbdde40099c09cca655454eabc7b02316efb0a9",
    2024: "65fa040101d784b2225eafea9a7a8e5660c5fca9f1b962e342cd81b9cb689f5c",
    2025: "16867d676338e22da82a84c72b667fe5764ff266a03d6e8086182de138a2c185",
}
SOURCE_DRIVE_IDS = {
    2023: "1qUZQb4Es6nL6Dq5zEBpzeF3ofwiVi8sc",
    2024: "1Dju6mOvLAuAsyHm3qiKRYxE6KxM5i6HT",
    2025: "1mXkKaVE7B1zoUY0u5xzhzqk-BDg1P8aM",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wave(path: Path, year: int) -> tuple[pd.DataFrame, dict]:
    spec = SPECS[year]
    source = pd.read_excel(path, sheet_name="Numeric", usecols=list(spec.values()))
    if source[spec["id"]].isna().any() or source[spec["id"]].duplicated().any():
        raise ValueError(f"{year}: missing or duplicate participant identifier")
    invalid = {}
    rules = {"use": [1, 2, 3, 4, 5, 9], "hesitation": [1, 2, 3, 4, 9], "A": [1, 2]}
    frame = pd.DataFrame({"id": source[spec["id"]].astype(str)})
    for target, allowed in rules.items():
        values = pd.to_numeric(source[spec[target]], errors="coerce")
        invalid[target] = int((values.notna() & ~values.isin(allowed)).sum())
        frame[target] = values.where(values.isin(allowed))
    frame["A"] = frame["A"].map({1: 1, 2: 0})
    frame["Y"] = frame["use"].isin([1, 2, 3, 4]).astype(int).where(frame["use"].notna())
    digest = sha256(path)
    audit = {
        "year": year,
        "file_name": path.name,
        "drive_id": SOURCE_DRIVE_IDS[year],
        "sheet": "Numeric",
        "sha256": digest,
        "sha256_matches_revision_source_manifest": digest == EXPECTED_SOURCE_HASHES[year],
        "source_n": int(len(frame)),
        "source_fields": spec,
        "invalid_nonmissing_codes": invalid,
        "missing_after_valid_code_filter": {k: int(frame[k].isna().sum()) for k in rules},
    }
    if not audit["sha256_matches_revision_source_manifest"]:
        raise ValueError(f"{year}: source file differs from the revision source manifest")
    return frame, audit


def make_transition(baseline: pd.DataFrame, followup: pd.DataFrame, year: int) -> pd.DataFrame:
    out = baseline[["id", "A", "hesitation", "Y"]].rename(columns={"Y": "H"}).merge(
        followup[["id", "Y"]], on="id", how="inner", validate="one_to_one"
    )
    out = out.dropna(subset=["Y"]).reset_index(drop=True)
    out["baseline_year"] = year
    out["outcome_year"] = year + 1
    out["Y"] = out["Y"].astype(int)
    return out


def id_set_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(frame.id)).encode()).hexdigest()


def wilson_interval(events: int, total: int) -> tuple[float, float]:
    z = float(norm.ppf(0.975))
    p = events / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return float(center - half), float(center + half)


def newcombe_difference(events1: int, total1: int, events0: int, total0: int) -> tuple[float, float]:
    p1, p0 = events1 / total1, events0 / total0
    lo1, hi1 = wilson_interval(events1, total1)
    lo0, hi0 = wilson_interval(events0, total0)
    difference = p1 - p0
    lower = difference - np.sqrt((p1 - lo1) ** 2 + (hi0 - p0) ** 2)
    upper = difference + np.sqrt((hi1 - p1) ** 2 + (p0 - lo0) ** 2)
    return float(lower), float(upper)


def prevalence_rows(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in samples.items():
        valid = frame.A.notna()
        count = int(frame.loc[valid, "A"].sum())
        n = int(valid.sum())
        lo, hi = wilson_interval(count, n)
        rows.append({
            "sample": name, "baseline_year": int(frame.baseline_year.iloc[0]),
            "outcome_year": int(frame.outcome_year.iloc[0]), "n": len(frame),
            "A_valid_n": n, "A_missing_n": int((~valid).sum()), "A_yes_n": count,
            "A_no_n": n - count, "A_yes_pct": 100 * count / n,
            "A_yes_pct_pointwise_95_low": 100 * float(lo),
            "A_yes_pct_pointwise_95_high": 100 * float(hi),
            "outcome_events": int(frame.Y.sum()), "outcome_pct": 100 * float(frame.Y.mean()),
        })
    return pd.DataFrame(rows)


def associations(samples: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assoc, cells = [], []
    for name, sample in samples.items():
        frame = sample.dropna(subset=["A", "Y"]).copy()
        tab = pd.crosstab(frame.A, frame.Y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        n0, n1 = int(tab.loc[0].sum()), int(tab.loc[1].sum())
        a, b, c, d = int(tab.loc[1, 1]), int(tab.loc[1, 0]), int(tab.loc[0, 1]), int(tab.loc[0, 0])
        if min(a, b, c, d) == 0:
            raise ValueError("A zero cell requires a separately specified interval method")
        log_or = np.log(a * d / (b * c))
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        or_ci = np.exp(log_or + np.array([-1, 1]) * norm.ppf(0.975) * se)
        rd = a / n1 - c / n0
        rd_ci = newcombe_difference(a, n1, c, n0)
        if not np.isclose(float(logit(a / n1) - logit(c / n0)), log_or, atol=1e-12):
            raise AssertionError("Cross-product OR and difference of logits disagree")
        assoc.append({
            "sample": name, "baseline_year": int(frame.baseline_year.iloc[0]),
            "outcome_year": int(frame.outcome_year.iloc[0]), "complete_n": len(frame),
            "A_yes_n": n1, "A_yes_outcome_events": a, "A_yes_outcome_pct": 100 * a / n1,
            "A_no_n": n0, "A_no_outcome_events": c, "A_no_outcome_pct": 100 * c / n0,
            "risk_difference_pp": 100 * rd,
            "risk_difference_pp_pointwise_95_low": 100 * float(rd_ci[0]),
            "risk_difference_pp_pointwise_95_high": 100 * float(rd_ci[1]),
            "unadjusted_odds_ratio": float(np.exp(log_or)),
            "unadjusted_OR_pointwise_95_low": float(or_ci[0]),
            "unadjusted_OR_pointwise_95_high": float(or_ci[1]),
            "phi_correlation": float(frame[["A", "Y"]].corr().iloc[0, 1]),
        })
        for av in [0, 1]:
            for yv in [0, 1]:
                cells.append({"sample": name, "A_experience": av, "outcome_any_use": yv, "count": int(tab.loc[av, yv])})
    return pd.DataFrame(assoc), pd.DataFrame(cells)


def overlap_table(full_dev: pd.DataFrame, full_eval: pd.DataFrame, restricted_dev: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_linked": np.ones(len(full_eval), dtype=bool),
        "hesitant": full_eval.hesitation.isin([1, 2]),
        "other": ~full_eval.hesitation.isin([1, 2]),
        "rare_or_never_hesitant": full_eval.hesitation.isin([3, 4]),
        "no_relevant_experience": full_eval.hesitation.eq(9),
    }
    rows = []
    for name, mask in groups.items():
        f = full_eval.loc[mask]
        shared = f.id.isin(full_dev.id)
        restricted_shared = f.id.isin(restricted_dev.id)
        rows.append({
            "evaluation_group": name, "evaluation_n": len(f),
            "overlap_with_full_development_n": int(shared.sum()),
            "overlap_with_full_development_pct": 100 * float(shared.mean()),
            "person_disjoint_from_full_development_n": int((~shared).sum()),
            "person_disjoint_from_full_development_pct": 100 * float((~shared).mean()),
            "overlap_with_restricted_development_n_for_reference_only": int(restricted_shared.sum()),
            "overlap_with_restricted_development_pct_for_reference_only": 100 * float(restricted_shared.mean()),
            "outcome_events": int(f.Y.sum()),
            "shared_outcome_events": int(f.loc[shared, "Y"].sum()),
            "disjoint_outcome_events": int(f.loc[~shared, "Y"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the focused JTAER minor-revision raw-data diagnostics; export aggregate data only.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waves, source_audit = {}, []
    for year in SPECS:
        waves[year], audit = read_wave(args.raw_dir / f"panel_{year}.xlsx", year)
        source_audit.append(audit)
    full_dev = make_transition(waves[2023], waves[2024], 2023)
    full_eval = make_transition(waves[2024], waves[2025], 2024)
    dev = full_dev.loc[full_dev.hesitation.isin([1, 2])].copy()
    eva = full_eval.loc[full_eval.hesitation.isin([1, 2])].copy()
    controls = {
        "source_2023_n": (len(waves[2023]), 4581), "source_2024_n": (len(waves[2024]), 4420),
        "source_2025_n": (len(waves[2025]), 4324), "restricted_development_n": (len(dev), 1565),
        "restricted_evaluation_n": (len(eva), 1520), "restricted_development_events": (int(dev.Y.sum()), 880),
        "restricted_evaluation_events": (int(eva.Y.sum()), 853), "restricted_overlap_n": (int(eva.id.isin(dev.id).sum()), 641),
        "full_development_n": (len(full_dev), 3895), "full_evaluation_n": (len(full_eval), 3741),
        "full_development_events": (int(full_dev.Y.sum()), 2215), "full_evaluation_events": (int(full_eval.Y.sum()), 2125),
        "full_overlap_n": (int(full_eval.id.isin(full_dev.id).sum()), 3333),
    }
    for key, (value, expected) in controls.items():
        if value != expected:
            raise AssertionError(f"{key}: {value} != {expected}")
    samples = {"restricted_development": dev, "restricted_evaluation": eva}
    prevalence = prevalence_rows(samples)
    association, cells = associations(samples)
    overlap = overlap_table(full_dev, full_eval, dev)
    paired = dev[["id", "A"]].merge(eva[["id", "A"]], on="id", suffixes=("_2023", "_2024"), validate="one_to_one")
    paired_counts = pd.crosstab(paired.A_2023, paired.A_2024).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    paired_rows = [{"A_2023": a, "A_2024": b, "count": int(paired_counts.loc[a, b])} for a in [0, 1] for b in [0, 1]]
    for name, table in {
        "A_prevalence": prevalence, "A_outcome_association": association, "A_outcome_2x2_counts": cells,
        "full_panel_group_overlap": overlap, "A_shared_participant_transitions": pd.DataFrame(paired_rows),
    }.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    manifest = {
        "purpose": "Focused additional diagnostics requested in minor revision; original model estimates are unchanged.",
        "source_audit": source_audit,
        "controls": {k: {"observed": v, "expected": e, "passed": v == e} for k, (v, e) in controls.items()},
        "cohort_id_set_hashes": {k: id_set_hash(v) for k, v in {**samples, "full_development": full_dev, "full_evaluation": full_eval}.items()},
        "A_prevalence": prevalence.to_dict("records"), "A_outcome_association": association.to_dict("records"),
        "full_panel_group_overlap": overlap.to_dict("records"), "A_paired_641": paired_rows,
        "interval_methods": {"prevalence": "Pointwise Wilson 95%", "risk_difference": "Pointwise Newcombe 95%", "odds_ratio": "Pointwise log-Wald 95%; cross-product OR validated against difference of outcome logits"},
        "scope": "Unweighted exploratory associations; separate-period estimates are descriptive. They do not estimate incremental multivariable prediction value or a causal effect. Overlap for the group comparison refers to the full linked development sample, not the restricted development cohort.",
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "script_sha256": sha256(Path(__file__)),
        "privacy": "All exported outputs are aggregate counts, estimates, source/schema metadata and nonreversible cohort-set hashes; no respondent rows or IDs are exported.",
    }
    (args.output_dir / "minor_diagnostics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(prevalence.to_string(index=False))
    print(association.to_string(index=False))
    print(overlap.to_string(index=False))
    print("All cohort controls and source hashes passed.")


if __name__ == "__main__":
    main()
