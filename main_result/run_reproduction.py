from pathlib import Path
import argparse, os, subprocess, sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(
        description="Run documented research analysis stages with authorized local data."
    )
    p.add_argument(
        "--stage",
        choices=[
            "base",
            "full_bootstrap",
            "added_models",
            "aggregate_diagnostics",
            "supplementary",
            "figures",
            "all",
        ],
        default="all",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not 1 <= a.jobs <= 8:
        p.error("--jobs must be between 1 and 8")
    if not __debug__:
        raise RuntimeError("Run without -O or -OO; analytical assertions are required.")
    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[name] = "1"
    for d in [
        "revision/tables",
        "revision/results",
        "revision/audit",
        "private_data",
        "analysis/round2",
        "JTAER_Figures/data",
        "JTAER_Figures/preview",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    stages = {
        "base": [
            ["source_audit.py"],
            ["codebook_audit.py"],
            ["questionnaire_audit.py"],
            ["refit_and_inference.py"],
            ["summarize_fixedfit_results.py"],
            ["run_model_sensitivity.py", "--jobs", str(a.jobs)],
            ["model_sensitivity_intervals.py"],
            ["full_panel_hesitancy_subgroups.py"],
            ["validate_model_sensitivity.py"],
            ["holdout_calibration.py", "holdout", "--jobs", str(a.jobs), "--repeats", "20"],
            ["holdout_calibration.py", "calibration", "--calibration-bootstrap", "500"],
            ["run_attribution.py"],
            ["validate_holdout_calibration.py"],
        ],
        "full_bootstrap": [
            ["run_selection_bootstrap.py", "--reps", "200", "--jobs", str(a.jobs)],
            ["validate_selection_bootstrap.py"],
            ["round2_run_selection_bootstrap.py", "--reps", "1000", "--jobs", str(a.jobs)],
            ["round2_validate_selection_bootstrap.py"],
            ["round2_reselection_report.py"],
        ],
        "added_models": [
            ["round2_boosting.py", "--mode", "all", "--jobs", str(min(a.jobs, 4))],
            ["round2_validate_boosting.py"],
            ["round2_numerical_traceability.py"],
        ],
        "aggregate_diagnostics": [["audit_interval_sensitivity.py"]],
        "supplementary": [],
        "figures": [
            ["plot_revision_figures.py", "--final"],
            ["round2_figures.py"],
            ["round2_bootstrap_figure.py"],
        ],
    }
    if (ROOT / "revision/code/diagnose_selection_and_harm.py").exists():
        stages["aggregate_diagnostics"].append(["diagnose_selection_and_harm.py"])
    if (ROOT / "revision/code/region_overlap_sensitivity.py").exists():
        stages["supplementary"].append(
            [
                "region_overlap_sensitivity.py",
                "--raw",
                str(ROOT / "private_raw"),
                "--package",
                str(ROOT),
                "--out",
                str(ROOT / "revision/results"),
                "--private",
                str(ROOT / "private_data/region_sensitivity"),
            ]
        )
    if (ROOT / "revision/code/adjacent_service_diagnostics.py").exists():
        stages["supplementary"].append(
            [
                "adjacent_service_diagnostics.py",
                "--raw-dir",
                str(ROOT / "private_raw"),
                "--output-dir",
                str(ROOT / "revision/results"),
            ]
        )
    if (ROOT / "revision/code/diagnose_selection_and_harm.py").exists():
        stages["supplementary"].append(
            [
                "diagnose_selection_and_harm.py",
                "--development",
                str(ROOT / "private_data/development.csv"),
                "--evaluation",
                str(ROOT / "private_data/evaluation.csv"),
            ]
        )
    if (ROOT / "revision/code/plot_submission_block_heatmap.py").exists():
        stages["figures"].append(["plot_submission_block_heatmap.py"])
    for stage in stages if a.stage == "all" else [a.stage]:
        for args in stages[stage]:
            command = [sys.executable, str(ROOT / "revision/code" / args[0]), *args[1:]]
            print("RUN", *command, flush=True)
            if not a.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
