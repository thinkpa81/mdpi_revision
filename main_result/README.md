# Final analysis materials

**Questionnaire Item Budgets and Incremental Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Temporal Evidence from a Korean Panel**

This research release contains executable analysis code and aggregate results for evaluating questionnaire item budgets and incremental predictive value through the temporal evaluation of self-reported AI-enabled consumer-service use in the Korean KISDI panel. It covers the original restricted-cohort models, expanded model searches, full-panel comparisons, complete-procedure bootstrap, and additional measurement and representation diagnostics.

## Contents and data access

- `revision/code`: executable analytical procedures.
- `revision/tables`: aggregate numerical tables from the completed analyses.
- `revision/results`: interval-construction, regularization, cohort, adjacent-service, and region-coding diagnostics supplied with this version.
- `revision/audit`: scientific protocols, source hashes, software versions, and analytical completion records.
- `original_results`: selected original analytical sources and aggregate results required by the reproducibility pipeline.
- `validate_public_release.py`: file-integrity and numerical reconstruction checks that need no private data.
- `run_reproduction.py`: ordered execution entry point for a fresh analysis.

Individual respondents, identifiers, fitted objects, checkpoints, and individual predictions are not redistributed. Obtain the 2023, 2024, and 2025 source workbooks through the KISDI AI Service User Policy Archive under its access and statistical-use conditions. Place authorized copies at `private_raw/2023_data.xlsx`, `private_raw/2024_data.xlsx`, and `private_raw/2025_data.xlsx`. For questionnaire/codebook audit stages, also supply the corresponding `YYYY_questionnaire.pdf` and `YYYY_codebook.xlsx` files. Korean literals retained in harmonization and source-schema audits denote original provider fields or questionnaire wording; they are not editorial instructions.

## Environment

The recorded model environment uses Python 3.12, NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0. Additional version pins are in `revision/requirements.txt` and `revision/requirements_round2.txt`; the scientific execution records identify versions actually used. Install dependencies into an isolated environment before running the pipeline. No script installs dependencies or downloads data automatically.

```bash
python -m pip install -r revision/requirements.txt -r revision/requirements_round2.txt
python validate_public_release.py
python run_reproduction.py --dry-run
```

The first check verifies the delivered bytes and independently reaggregates all 8,000 stored complete-procedure records. It is a numerical consistency check, not new model fitting. Preserve a copy of the checked repository snapshot before recomputation: fresh runs deliberately write their own results and provenance, so the original file-integrity manifest will subsequently differ.

## Fresh analysis

Run in a separate clean working copy containing the authorized source files. Existing private caches from different source bytes, model grids, code, seeds, or library versions must not be reused. Task-cache and bootstrap-checkpoint checks reject mismatched or incomplete receipts.

```bash
python run_reproduction.py --stage base --jobs 4
python run_reproduction.py --stage full_bootstrap --jobs 4
python run_reproduction.py --stage added_models --jobs 4
python run_reproduction.py --stage aggregate_diagnostics
python run_reproduction.py --stage supplementary --jobs 4
python run_reproduction.py --stage figures
```

The base stage reconstructs restricted and full linked cohorts, refits the original models, computes paired benefit/harm inference, fits expanded algorithms, and evaluates within-period holdouts and calibration. The full-bootstrap stage first computes the historical 200-repetition sequence and then extends the same seed sequence to 1,000 repetitions under verified input/code/configuration bindings. The supplied historical 200 records are exactly the first 200 records of the completed sequence for each procedure; they are not an additional independent sample. A new end-to-end run refits the sequence instead of assuming access to private historical checkpoint files.

The complete-procedure bootstrap applies to budgets 1 and 4, the original ridge/HGB grids, both history-availability conditions, and the restricted whole evaluation cohort. Other analyses condition on the fitted models unless their protocols explicitly state otherwise. Percentile intervals are the primary summaries. Basic intervals and configuration-conditioned summaries describe construction sensitivity and regularization shifts; they do not establish nominal coverage after discrete selection. Subgroup differences and adjacent-service contrasts are observational diagnostics, not identified causal effects of privacy hesitation, participant overlap, or wording.

The additional diagnostics can also be run individually with `--help`; their scripts record inputs, seeds, model choices, and output schema. The `supplementary` stage writes participant-level intermediate files only inside `private_data`.

Figure code writes PNG, PDF, and SVG versions plus source CSVs to `JTAER_Figures`. The repository-level `../JTAER_Figures/` directory supplies the manuscript-numbered exports and aggregate plot data. Reproducing the analytical charts does not require editorial response files or document-generation scripts.

## Result provenance and current scope

The current implementation is in `revision/code/`. The retained `original_results/` files support source harmonization, original-fit checks, nested cross-validation displays, threshold-specific results in Supplementary Table S25, and numerical traceability. They remain part of the evidence for the final paper. The historical 200 bootstrap repetitions and their validation records support the documented extension to 1,000 repetitions and remain available.

Earlier implementation copies from `historical_source/` are archived at [the manuscript-cited snapshot](https://github.com/thinkpa81/mdpi_revision/tree/af06ecda65c0f5768cbe2b3449bb46ebf229edca/main_result/historical_source). They are not imported by the current execution pipeline and have been removed from the current tree. The same snapshot preserves the original package manifest, normalization records, and generating-code history. Current files and hashes are listed in `PUBLIC_RELEASE_MANIFEST.json` and `SHA256SUMS.json`.

The original analysis release is commit `f3b0a46b86f8ab0a7e6093f17105d0f38d281824` at https://github.com/thinkpa81/mdpi. The preceding public revision is `research_release_20261021/` at commit `34f4172e3bebdb97fe5d5044dd51b237a7c82881` in that repository. These archived sources distinguish retained historical outputs from newly executed analyses.

The final repository alignment changes documentation, figure captions, metadata, and redundant historical copies. Current analysis code, aggregate numerical results, figure-data CSVs, and PNG/PDF/SVG exports are preserved. Recorded execution receipts describe their original runs; they are not claims that all analyses were rerun during this alignment. See `../FINAL_ALIGNMENT_CHECK.json` for the final document identities, removals, and validation results, and `../SOURCE_RECHECK_20261024.json` for the separate source reconstruction and adjacent-service recheck.
