# JTAER minor-revision reproducibility addendum

Manuscript: jtaer-4619767. Prepared 1 November 2026.

This directory is an additive supplement to `thinkpa81/mdpi_revision`, whose
previous numerical analysis release is commit `af06ecda65c0f5768cbe2b3449bb46ebf229edca`. The directory
`minor_revision_20261101_v1/` preserves the earlier analysis and adds the diagnostics
and presentation checks below. Cite the actual Git commit containing the files used.

## Contents

| Path | Purpose |
|---|---|
| `analysis/` | Focused generative-AI experience and full-panel group-overlap diagnostics |
| `analysis/instruments/` | Core question, response-code and surrounding-context audit |
| `bootstrap_verification/` | Exact archived fitting code, aggregate records and common-environment verification |
| `qa/` | Independent raw-workbook verification, public-package checker, seed catalogue and aggregate audit receipts |
| `figures/` and `code/` | Figure 7 with (a)–(d) panel labels, aggregate inputs, regeneration code and pixel-scope verification |
| `requirements.txt` | Exact package versions used in the verification environment |
| `SHA256SUMS` | Integrity hashes for this release |

This public addendum contains code and aggregate outputs. It excludes the manuscript,
review correspondence, participant-level raw data, identifiers, fitted prediction arrays,
model caches and survey-instrument source documents.

## Environment

Use Python 3.12.14. From this directory:

```bash
python -m pip install -r requirements.txt
```

Versions: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0,
joblib 1.5.3, threadpoolctl 3.6.0 and openpyxl 3.1.5. Figure generation uses
Matplotlib 3.10.8; image verification uses Pillow 12.3.0.

## Focused diagnostics

Obtain the authorized KISDI Numeric-sheet workbooks separately. Use filenames
`panel_2023.xlsx`, `panel_2024.xlsx` and `panel_2025.xlsx` in a private input directory.
Expected SHA-256 values and source-variable definitions are in
`analysis/README_DIAGNOSTICS.md`.

```bash
python analysis/minor_raw_diagnostics.py --raw-dir /path/to/authorized_inputs --output-dir /path/to/new_aggregate_outputs
```

To repeat the independent verification, place the same workbooks under the ignored
`analysis/raw/` directory and run `python qa/verify_minor_aggregates.py`.
The included audit passed 115 comparisons, including 13 historical sample/event/overlap
controls. It imports no production analysis functions and exports aggregate checks only.

The two restricted cohorts contain 1,565 and 1,520 people, with 641 shared participants.
A experience prevalence is 193/1,565 (12.33%) and 458/1,520 (30.13%). The full-panel
overlap diagnostic instead uses the 3,895-person full development cohort: 1,369/1,520
hesitant respondents (90.07%) and 1,964/2,221 other respondents (88.43%) overlap.
The original model fits, comparison families and performance results remain unchanged.

## Bootstrap verification without raw data

From `bootstrap_verification/`:

```bash
python code/verify_recomputed_segment.py \
  --new-tables exact_original_source/main_result/revision/tables \
  --historical-tables historical_tables \
  --historical-manifest historical_audit/round2_reselection_bootstrap_manifest.json \
  --environment rerun_environment.json --out /path/to/new_verification_outputs
```

The first 200 full-process bootstrap draws were independently regenerated using the
unchanged archived fitting code, original seed sequence and the software versions
documented for the final 800 draws. All 1,600 aggregate records and all eight point
estimates matched exactly. Combining these recomputed draws with the unchanged final
800 draws left every published interval and frequency unchanged. The original 200-draw
manifest lacked a separate software-version receipt; this addendum documents the
recomputation instead of claiming unsupported historical version identity.

`bootstrap_verification/README.md` gives the exact raw-data reconstruction and fitting
commands, source hashes and the distinct seed definitions. The two highlighted seed values refer to different calculations: 20261020 for the
original fixed-prediction comparison families, and 2026102002 for Figure 2 pointwise
absolute-loss intervals. Other families use the analysis-specific seeds listed in
`qa/public_seed_catalog.json`. No seed or result was changed.

## Figure 7 presentation verification

`figures/README_Figure_7.md` documents the generation and inspection commands.
Only the panel prefixes changed from A–D to (a)–(d). All 84 source rows are
byte-identical; all image pixels outside the panel-title bands are unchanged.
The seven outlined temporal cells remain six for D and one for A.

## Verify this addendum

```bash
python qa/verify_public_addendum.py
```

This command checks the complete file set, SHA-256 values, Python/JSON syntax,
public CSV structure, exact aggregate bootstrap comparison and Figure 7 pixel scope.
It does not read raw data or refit models. Use `sha256sum -c SHA256SUMS` for a
standard-library-independent file-integrity check on systems with that utility.
The repository-level `verify_release.py` additionally checks the complete repository.
