#!/usr/bin/env python3
"""Verify independently recomputed draws 1:200, then assemble aggregate draws 1:1000.

This compares aggregate model-selection/performance records, never respondent data.
Read all floating-point CSV values with round-trip precision; no rounding occurs.
The existing 201:1000 segment is preserved. Exit unsuccessfully on any discrepancy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["replicate", "algorithm", "history_available", "budget"]
CONDITION_KEYS = KEYS[1:]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    # Literal JSON "null" is a valid selected_config value for an intercept-only
    # benchmark. Preserve it as text rather than treating it as a missing cell.
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def compare(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> dict:
    assert list(a.columns) == list(b.columns), "Different table schemas"
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    assert a.shape == b.shape, "Different table shapes"
    columns = {}
    for name in a:
        eq = (a[name] == b[name]) | (a[name].isna() & b[name].isna())
        record = {"exact_equal": bool(eq.all()), "unequal_cells": int((~eq).sum())}
        if pd.api.types.is_numeric_dtype(a[name]) and not pd.api.types.is_bool_dtype(a[name]):
            difference = np.abs(a[name].to_numpy(dtype=float) - b[name].to_numpy(dtype=float))
            record["max_absolute_difference"] = float(np.nanmax(difference))
        columns[name] = record
    return {"rows": len(a), "columns": len(a.columns), "all_cells_exact_equal": all(v["exact_equal"] for v in columns.values()), "by_column": columns}


def summarize(boot: pd.DataFrame, point: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, frequencies = [], []
    for key, group in boot.groupby(CONDITION_KEYS):
        point_row = point[(point.algorithm == key[0]) & (point.history_available == key[1]) & (point.budget == key[2])].iloc[0]
        n = len(group)
        record = dict(zip(CONDITION_KEYS, key))
        record.update(replicates=n, point_subset=point_row.selected_subset, point_items=int(point_row.selected_items))
        for column in ["temporal_log_loss", "delta_vs_benchmark", "delta_vs_full", "roc_auc"]:
            record[column + "_point"] = float(point_row[column])
            record[column + "_bootstrap_mean"] = float(group[column].mean())
            record[column + "_lower_025"] = float(group[column].quantile(.025))
            record[column + "_upper_975"] = float(group[column].quantile(.975))
            record[column + "_bootstrap_sd"] = float(group[column].std(ddof=1))
        record["fraction_delta_vs_benchmark_positive"] = float((group.delta_vs_benchmark > 0).mean())
        summaries.append(record)
        for subset, subgroup in group.groupby("selected_subset"):
            rate = len(subgroup) / n
            frequencies.append(dict(**dict(zip(CONDITION_KEYS, key)), selected_subset=subset, count=len(subgroup), frequency=rate, monte_carlo_se=float(np.sqrt(rate * (1-rate) / n))))
    return pd.DataFrame(summaries), pd.DataFrame(frequencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tables", type=Path, required=True)
    parser.add_argument("--historical-tables", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    new_path = args.new_tables / "reselection_bootstrap_replicates.csv"
    old_path = args.historical_tables / "reselection_bootstrap_replicates.csv"
    all_path = args.historical_tables / "round2_reselection_bootstrap_replicates.csv"
    new_point_path = args.new_tables / "reselection_point_estimates.csv"
    old_point_path = args.historical_tables / "round2_reselection_point_estimates.csv"
    new, old, historical_all = read(new_path), read(old_path), read(all_path)
    point, old_point = read(new_point_path), read(old_point_path)
    assert set(new.replicate) == set(range(1, 201)) and len(new) == 1600
    assert set(historical_all.replicate) == set(range(1, 1001)) and len(historical_all) == 8000
    new_vs_old = compare(new, old, KEYS)
    new_vs_retained = compare(new, historical_all[historical_all.replicate <= 200], KEYS)
    point_comparison = compare(point, old_point, KEYS)
    assert new_vs_old["all_cells_exact_equal"]
    assert new_vs_retained["all_cells_exact_equal"]
    assert point_comparison["all_cells_exact_equal"]
    combined = pd.concat([new, historical_all[historical_all.replicate > 200]], ignore_index=True)
    # Preserve within-replicate condition ordering from the original fitting script.
    combined = combined.sort_values("replicate", kind="stable").reset_index(drop=True)
    complete_comparison = compare(combined, historical_all, KEYS)
    assert complete_comparison["all_cells_exact_equal"]
    intervals, frequencies = summarize(combined, point)
    summary_comparison = compare(intervals, read(args.historical_tables / "round2_reselection_bootstrap_intervals.csv"), CONDITION_KEYS)
    frequency_comparison = compare(frequencies, read(args.historical_tables / "round2_reselection_bootstrap_frequencies.csv"), CONDITION_KEYS + ["selected_subset"])
    assert summary_comparison["all_cells_exact_equal"]
    assert frequency_comparison["all_cells_exact_equal"]
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "minor_reselection_recomputed_200.csv": new,
        "minor_reselection_bootstrap_replicates.csv": combined,
        "minor_reselection_point_estimates.csv": point,
        "minor_reselection_bootstrap_intervals.csv": intervals,
        "minor_reselection_bootstrap_frequencies.csv": frequencies,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out / name, index=False)
    byte_reference = {
        "minor_reselection_recomputed_200.csv": old_path,
        "minor_reselection_bootstrap_replicates.csv": all_path,
        "minor_reselection_point_estimates.csv": old_point_path,
        "minor_reselection_bootstrap_intervals.csv": args.historical_tables / "round2_reselection_bootstrap_intervals.csv",
        "minor_reselection_bootstrap_frequencies.csv": args.historical_tables / "round2_reselection_bootstrap_frequencies.csv",
    }
    byte_identity = {name: (args.out / name).read_bytes() == reference.read_bytes() for name, reference in byte_reference.items()}
    assert all(byte_identity.values()), "Numerically equal output differs in CSV serialization"
    environment = json.loads(args.environment.read_text())
    historical_manifest = json.loads(args.historical_manifest.read_text())
    recorded_versions = historical_manifest["package_versions"]
    version_matches = {name: environment["software"]["scikit-learn" if name == "sklearn" else name] == version for name, version in recorded_versions.items()}
    assert all(version_matches.values()), "Rerun versions differ from the documented final-800 versions"
    assert historical_manifest["fitting_script_sha256"] == environment["code_sha256"]["run_selection_bootstrap.py"]
    audit = {
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Remove missing per-batch software-version evidence for the originally retained first 200 draws by independently regenerating them using the fitting script, inputs, seed sequence, and software versions documented for draws 201:1000.",
        "historical_software_limitation": "The original 200-draw manifest does not contain a separate package-version receipt. This verification does not assert unsupported historical version identity.",
        "final_segment_origin": {"replicates_1_to_200": "Recomputed for this minor revision in the documented final-800 environment", "replicates_201_to_1000": "Existing documented final-800 results, unchanged"},
        "seed": {"base": 202610200, "replicate_seed_rule": "base + replicate_index", "recomputed_indices": [1, 200], "retained_indices": [201, 1000]},
        "environment": environment,
        "documented_final_800_package_versions": recorded_versions,
        "rerun_versions_match_documented_final_800": version_matches,
        "final_800_manifest_sha256": sha(args.historical_manifest),
        "max_absolute_numeric_difference_for_recomputed_200": max(v.get("max_absolute_difference", 0) for v in new_vs_old["by_column"].values()),
        "new_200_vs_original_200": new_vs_old,
        "new_200_vs_retained_segment_of_1000": new_vs_retained,
        "point_estimates": point_comparison,
        "combined_1000_vs_previous_1000": complete_comparison,
        "recomputed_intervals_vs_previous_1000": summary_comparison,
        "recomputed_frequencies_vs_previous_1000": frequency_comparison,
        "input_aggregate_sha256": {p.name + ("_recomputed" if p.parent == args.new_tables else "_historical"): sha(p) for p in [new_path, old_path, all_path, new_point_path, old_point_path]},
        "output_sha256": {name: sha(args.out / name) for name in outputs},
        "output_csv_byte_identical_to_historical": byte_identity,
        "verification_script_sha256": sha(Path(__file__)),
        "verification_runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "individual_level_data_in_outputs": False,
        "published_results_changed": False,
    }
    (args.out / "minor_reselection_software_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "recomputed_draws": 200, "recomputed_aggregate_rows": len(new), "combined_draws": 1000, "combined_aggregate_rows": len(combined), "all_outputs_numerically_exact": True, "output_directory": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
