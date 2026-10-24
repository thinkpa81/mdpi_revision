# Questionnaire collection and incremental prediction — v4

This research release contains executable analysis code and aggregate results for the temporal evaluation of self-reported AI-enabled consumer-service use in the Korean KISDI panel. It covers the original restricted-cohort models, expanded model searches, full-panel comparisons, complete-procedure bootstrap, and additional measurement and representation diagnostics.

## Contents and data access

- `revision/code`: executable analytical procedures.
- `revision/tables`: aggregate numerical tables from the completed analyses.
- `revision/results`: interval-construction, regularization, cohort, adjacent-service, and region-coding diagnostics supplied with this version.
- `revision/audit`: scientific protocols, source hashes, software versions, and analytical completion records.
- `original_results`: selected original analytical sources and aggregate results required by the reproducibility pipeline.
- `historical_source`: earlier analytical procedures, with comments and docstrings removed and formatting standardized for this release. Original byte hashes and the transformation checks are retained in the release manifest; these files are not byte-identical archival copies.
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

The first check verifies the delivered bytes and independently reaggregates all 8,000 stored complete-procedure records. It is a numerical consistency check, not new model fitting. Preserve a copy of the delivered archive before recomputation: fresh runs deliberately write their own results and provenance, so the original file-integrity manifest will subsequently differ.

## Fresh analysis

Run in a clean extracted project directory containing the authorized source files. Existing private caches from different source bytes, model grids, code, seeds, or library versions must not be reused. Task-cache and bootstrap-checkpoint checks reject mismatched or incomplete receipts.

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

Figure code writes PNG, PDF, and SVG versions plus source CSVs to `JTAER_Figures`. The separate figure archive accompanying the manuscript supplies the manuscript-numbered exports. Reproducing the analytical charts does not require editorial response files or document-generation scripts.

## Result provenance

The original repository release is identified by commit `f3b0a46b86f8ab0a7e6093f17105d0f38d281824` at https://github.com/thinkpa81/mdpi. Current source files and delivered results are identified by `PUBLIC_RELEASE_MANIFEST.json` and `SHA256SUMS.json`. Historical execution records preserve generating-code hashes; current execution code can include validated cache or numerical-acceptance safeguards. This distinction does not imply that every stored analysis was newly rerun with the current source.

## Version 5 scope

This version draws significant temporal heatmap cells along their full boundaries. The same seven cells are outlined, using the existing 56-contrast simultaneous 95% intervals. All stored CSV files and scientific estimates are unchanged from version 4. The adjacent-service analyses remain diagnostics of observed response changes; they do not establish longitudinal measurement equivalence or isolate a wording effect. Region recoding remains post hoc.

The historical_source directory contains normalized executable copies of earlier procedures, not byte-identical archived originals. The original source hashes and normalization records are retained in the manifest. Original bytes remain available in the preceding public release. Python files retain command-line help and contain no comments or docstrings.

This package accompanies the manuscript as supplementary analysis material. Its source and aggregate outputs are identified by the included file manifest. The preceding public revision is available at commit `34f4172e3bebdb97fe5d5044dd51b237a7c82881`, under `research_release_20261021/`; that commit identifies version 3. Version 5 has no assigned repository commit before its separate publication. Record the actual commit after upload.
