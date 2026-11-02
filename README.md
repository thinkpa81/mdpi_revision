# Questionnaire budgets and temporal prediction of AI service use

**Questionnaire Item Budgets and Incremental Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Temporal Evidence from a Korean Panel**

Soonho Jang and Eung-Kyo Suh, Dankook University.

This is the canonical repository for the current reproducibility materials.
The version identifier is `v2.0-minor-revision`; the GitHub release records the exact commit for this snapshot.
See [release information](reproducibility/RELEASE.json), [release notes](RELEASE_NOTES.md), and the [functional index](reproducibility/INDEX.json).

| Purpose | Location |
| --- | --- |
| Main execution entry point | `main_result/run_reproduction.py` |
| Main analysis code and aggregate tables | `main_result/revision/code/`, `main_result/revision/tables/` |
| Additional A and participant-overlap diagnostics | `minor_revision_20261101_v1/analysis/` |
| Bootstrap replay and software verification | `minor_revision_20261101_v1/bootstrap_verification/` |
| Analysis-specific seed catalog | `minor_revision_20261101_v1/qa/public_seed_catalog.json` |
| Figures and source data | `JTAER_Figures/`; updated Figure 7 in `minor_revision_20261101_v1/figures/` |
| File integrity and syntax check | `python verify_release.py` |

Existing analytical paths are retained to keep executable code and provenance links stable. Dated directories identify historical analysis bundles; the Git commit and release tag identify the complete repository snapshot. This documentation update does not refit models or change analytical code, aggregate estimates, or figure data.

The earlier `thinkpa81/mdpi` repository remains a historical archive. Its initial public code snapshot is commit `f3b0a46b86f8ab0a7e6093f17105d0f38d281824`; the later `research_release_20261021` bundle is at `34f4172e3bebdb97fe5d5044dd51b237a7c82881`. These are separate historical states, not two targets of a single tag. New work is maintained here.

The public materials comprise code, aggregate outputs, figures, and provenance records. KISDI manages source microdata under its access and statistical-use conditions. Manuscripts, reviewer correspondence, participant records, identifiers, and individual predictions are outside this public release.

## Detailed reproduction instructions and analysis provenance

## Contents

| Path | Purpose |
| --- | --- |
| [minor_revision_20261101_v1](minor_revision_20261101_v1/README.md) | Current additive diagnostics and reproducibility checks |
| [minor_revision_20261101_v1/qa/public_seed_catalog.json](minor_revision_20261101_v1/qa/public_seed_catalog.json) | Seeds, their specific comparison families and aggregate interpretation checks |
| [minor_revision_20261101_v1/figures](minor_revision_20261101_v1/figures) | Figure 7 with lowercase panel labels, plotting code, aggregate data and visual comparison receipt |
| [main_result/revision/code](main_result/revision/code) | Reconstruction, model fitting, inference, bootstrap, sensitivity and original plotting code |
| [main_result/revision/tables](main_result/revision/tables) | Aggregate model results and bootstrap/holdout summaries |
| [main_result/revision/results](main_result/revision/results) | Interval, selection, overlap, regional and adjacent-service diagnostics |
| [main_result/revision/audit](main_result/revision/audit) | Source hashes, protocols, recorded versions and execution receipts |
| [main_result/original_results](main_result/original_results) | Original code and aggregate evidence retained for numerical comparisons and Supplementary Table S25 |
| [JTAER_Figures](JTAER_Figures) | Earlier publication figures, aggregate plot data and crosswalk; Figure 7 is superseded for presentation by the new export above |
| [FILE_MANIFEST_SHA256.json](FILE_MANIFEST_SHA256.json) | Hashes of all current public files, except the manifest itself |

Supplementary Table S26 maps the original analysis groups to scripts and outputs. Existing relative paths are retained. Dated alignment receipts, including [EDITORIAL_ALIGNMENT_CHECK_20261025.json](EDITORIAL_ALIGNMENT_CHECK_20261025.json), describe the document snapshots inspected at those dates; they are historical records rather than validation receipts for later documents. Submission documents are not distributed here.

## Verify the public materials

```bash
python verify_release.py
python -m pip install -r minor_revision_20261101_v1/requirements.txt
cd minor_revision_20261101_v1
python qa/verify_public_addendum.py
```

The first command checks repository hashes and Python/JSON syntax without running analytical code. The addendum verifier checks its checksums, allowed file types, aggregate CSV structure and the exact aggregate bootstrap comparison. It needs no raw data. Its scope does not include new model fitting.

The separate original numerical validator can also be run from the repository root:

```bash
python -m pip install -r main_result/revision/requirements.txt
cd main_result
python validate_public_release.py
python run_reproduction.py --dry-run
```

The original validator reaggregates 8,000 procedure-level bootstrap records, checks the initial 200-repetition segment and reconstructs interval and selection-frequency summaries. These rows describe analysis repetitions, not individual respondents.

## Reproduce with authorized source data

Obtain the 2023–2025 workbooks from the KISDI AI Service User Policy Archive under the provider's access conditions. Place the authorized `YYYY_data.xlsx`, `YYYY_codebook.xlsx` and `YYYY_questionnaire.pdf` files in `main_result/private_raw/`. Follow [main_result/README.md](main_result/README.md) for dependencies, scope and execution order. Additional boosting dependencies are listed in `main_result/revision/requirements_round2.txt`.

Run the original pipeline in a separate working copy because fresh execution replaces results and execution receipts. Use Python without `-O` or `-OO` so analytical assertions remain active. Focused addendum diagnostics have a separate command and input-name convention documented in [its README](minor_revision_20261101_v1/README.md).

The initial 200 full-process bootstrap draws were independently regenerated from the exact archived fitting code using the package versions documented for the final 800 draws. All 1,600 aggregate records and all eight point estimates matched exactly. Combining the regenerated draws with the retained final 800 draws left the published intervals and frequencies unchanged. The earlier 200-draw manifest did not include a separate software-version receipt, so the addendum documents the recomputation rather than asserting historical software identity without evidence.

## Interpretation and seed scope

Seed 20261020 applies to the original fixed-prediction comparison families. Figure 2 uses seed 2026102002 for pointwise intervals around absolute log loss. Attribution, regional recoding and other sensitivity analyses have separate seeds listed in the [seed catalogue](minor_revision_20261101_v1/qa/public_seed_catalog.json). No seed or numerical result was changed when clarifying these scopes.

This is a retrospective exploratory secondary analysis. Additional tuning and item selection used development data. Analyses that revisit the evaluation cohort remain exploratory. The full-process bootstrap covers item budgets 1 and 4 using the original ridge/HGB grids and the restricted whole evaluation cohort; other outputs retain their stated inference scopes. The new Figure 7 export changes panel letters only and retains all values, axes, colors and significant-cell outlines.

## Public-data boundary and citation

The public material contains code, metadata, aggregate statistics and figures. It excludes respondent records, identifiers, individual predictions, participant-specific resampling multiplicities, fold assignments, fitted objects, private caches, source workbooks, full survey instruments, submission documents and review correspondence. Obtain provider-controlled data separately. Ignore rules are safeguards for local Git use; the same boundary applies to uploads through other interfaces.

For reproducible citation, use the actual Git commit containing the files used. This README deliberately does not embed its own future commit hash. The prior numerical snapshot remains available at [`af06ecda65c0f5768cbe2b3449bb46ebf229edca`](https://github.com/thinkpa81/mdpi_revision/commit/af06ecda65c0f5768cbe2b3449bb46ebf229edca). The original analysis is archived in [thinkpa81/mdpi](https://github.com/thinkpa81/mdpi) at commit `f3b0a46b86f8ab0a7e6093f17105d0f38d281824`; the earlier research release is under `research_release_20261021/` at commit `34f4172e3bebdb97fe5d5044dd51b237a7c82881`.

No additional software license or provider-data redistribution rights are granted by this addendum.
