# Full-process bootstrap software verification

This supplementary reproducibility bundle documents the software-version check requested in the minor revision. It contains aggregate bootstrap records, source code, and audit metadata. It contains no raw survey rows, participant identifiers, or fitted prediction arrays.

The original 200-draw manifest records the fitting-script hash, input-data hashes, seed, folds, and model grid, but does not record a separate software-version receipt. The 1,000-draw extension records the versions used when generating draws 201–1,000. Historical software identity for the first 200 draws should therefore not be asserted solely from the manifests.

To remove this uncertainty from the final analysis, draws 1–200 and the eight original point estimates were independently recomputed using the exact original fitting-script bytes, exactly reconstructed input CSVs, the original seed sequence, and the documented package versions used for the final 800 draws. The aggregate comparison and the combined 1,000-draw audit are in `results/minor_reselection_software_verification.json`. The verified combined file uses these recomputed first 200 draws and the unchanged final 800 draws.

Verification passed for all 1,600 recomputed aggregate rows and all eight point-estimate rows. Every numeric value and selected configuration matched exactly (maximum absolute numeric difference: 0). The recomputed 200-draw CSV, combined 1,000-draw CSV, point-estimate CSV, interval summaries, and selection-frequency summaries are all byte-identical to their historical counterparts. No published value changed.

## Source and software

- Prior public release: `thinkpa81/mdpi_revision`, commit `af06ecda65c0f5768cbe2b3449bb46ebf229edca`.
- Exact historical source recovered from the authors' `main_result.zip` archive, Google Drive file ID `1A0HaRB0sN-mKjx-sN1Ra0Vo-uWVdpby2`. The archived script bytes match the historical manifests. Source files normalized in the repository can have different byte hashes even when their fitting logic is unchanged; the exact archived files are therefore retained here.
- Fitting script SHA-256: `d2b441fe7ecaa494f768f5dbdb0c3a86ebb4302216e40851bf9a852acf1904d0`.
- Model-engine SHA-256: `aaf3263877a14dbd3876a7ddc25f72bbca98f7d684d19f5035be19a6289dc2da`.
- Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; joblib 1.5.3. SciPy 1.17.0 and threadpoolctl 3.6.0 also match the shared historical runtime record. Raw XLSX reconstruction uses openpyxl 3.1.5.
- Replicate seed: `202610200 + replicate_index`. Recomputed indices: 1–200; retained indices: 201–1,000.

`rerun_environment.json` records the reconstructed cohort hashes, source-code hashes, and package versions. `historical_audit/` preserves the relevant existing receipts without modifying them.

## Repeating the aggregate verification

From this directory, with Python 3.12.14 and `requirements_verification.txt` installed:

```bash
python code/verify_recomputed_segment.py \
  --new-tables exact_original_source/main_result/revision/tables \
  --historical-tables historical_tables \
  --historical-manifest historical_audit/round2_reselection_bootstrap_manifest.json \
  --environment rerun_environment.json \
  --out results
```

The supplied `results/` files are the output from this verification. The recomputed aggregate CSV and point estimates are also copied under `exact_original_source/main_result/revision/tables/`, so the aggregate verification can be repeated without raw data. The script checks every cell with round-trip floating-point parsing, verifies all selected subsets and hyperparameters, reconstructs the 1,000-draw summaries, and fails on any discrepancy. No tolerance or rounding is used for the equality tests.

## Repeating the model fits with authorized raw files

Raw files are not distributed. To repeat the fit verification, make a separate working copy of `exact_original_source/main_result/`, place the authorized Numeric-sheet XLSX files at `private_raw/2023_data.xlsx`, `private_raw/2024_data.xlsx`, and `private_raw/2025_data.xlsx`, and run:

```bash
python revision/code/source_audit.py
python revision/code/run_selection_bootstrap.py --reps 200 --jobs 8
```

The historical script reconstructs the cohorts and regenerates the original 200 bootstrap draws. It writes private cohort files and checkpoints beneath the working copy; these must remain outside any public release. To recompute all 1,000 draws from scratch with the same fitting script, use `--reps 1000` in a fresh working copy. Aggregate verification should then be run against the supplied historical tables.

## Distinct seeds in the paper

| Seed | Calculation | Uncertainty target |
|---|---|---|
| 20261020 | 5,000 evaluation-participant bootstrap draws in `refit_and_inference.py` | Original fixed-prediction item-budget, benchmark-only sensitivity and full-information log-loss contrast families |
| 2026102002 | 5,000 evaluation-participant bootstrap draws in `round2_figures.py`, `frontier()` | Figure 2 pointwise percentile intervals for absolute temporal log loss, conditional on fixed fitted predictions |
| 202610200 + index | 1,000 participant-union bootstrap draws in `run_selection_bootstrap.py` | Complete budget-1 and budget-4 tuning, selection, and fitting procedure |

The first two seeds refer to separate analyses and are not a typographical inconsistency. Figure 2 intervals are pointwise intervals for absolute loss; they are not the simultaneous bounds for loss contrasts.

Other contrast families use analysis-specific seeds. See [`../qa/public_seed_catalog.json`](../qa/public_seed_catalog.json) for attribution, regional recoding and expanded-model seeds. No seed or result was changed.
