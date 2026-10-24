# JTAER major-revision research materials

**Questionnaire Collection Volume and Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Evidence from a Korean Panel**

Soonho Jang and Eung-Kyo Suh, Dankook University.

This repository publishes the analytical materials accompanying the October 2026 JTAER revision. It contains the extracted contents of the supplied `main_result.zip` and `JTAER_Figures.zip`: analysis code, aggregate results, execution records, and manuscript-numbered figures. The 234 source files are preserved byte for byte. The repository-level documentation and verification files were added on 24 October 2026.

[한국어 안내](README_KO.md)

## Repository contents

| Path | Contents |
| --- | --- |
| [main_result/revision/code](main_result/revision/code) | Current analysis procedures: reconstruction, original model refits, inference, resampling, model/population sensitivity, measurement diagnostics, and plotting |
| [main_result/revision/tables](main_result/revision/tables) | Aggregate model results, bootstrap summaries and replicate-level aggregate statistics |
| [main_result/revision/results](main_result/revision/results) | Interval-method, selection, participant-overlap, regional-coding, and adjacent-service diagnostics |
| [main_result/revision/audit](main_result/revision/audit) | Source hashes, protocols, recorded versions, and historical execution receipts |
| [main_result/original_results](main_result/original_results) | Selected earlier code and aggregate outputs required by the revision pipeline |
| [main_result/historical_source](main_result/historical_source) | Earlier procedures retained to document result provenance; see the supplied manifest for normalization details |
| [JTAER_Figures/Main](JTAER_Figures/Main) | Main Figures 1–7, each in PNG, PDF and SVG |
| [JTAER_Figures/Supplementary](JTAER_Figures/Supplementary) | Supplementary Figures S1–S4, each in PNG, PDF and SVG |
| [JTAER_Figures/data](JTAER_Figures/data) | Aggregate figure data and workflow-diagram text |
| [JTAER_Figures/FIGURE_CROSSWALK.csv](JTAER_Figures/FIGURE_CROSSWALK.csv) | Manuscript numbers, analytical source names, and captions |
| [RELEASE_VALIDATION.json](RELEASE_VALIDATION.json) | Checks performed when preparing this GitHub release |
| [FILE_MANIFEST_SHA256.json](FILE_MANIFEST_SHA256.json) | Integrity manifest for the released files |

The archive layout is retained because code and recorded provenance refer to these relative paths. Earlier inputs under `original_results` and `historical_source` support the current results and are not a second recommended execution pipeline. Manuscript Word files, marked drafts, correspondence, original ZIPs, and redundant copies of the entire earlier repository are outside this release.

## Verify without private data

Python 3.12 is the recorded analysis interpreter. The repository integrity check uses only the Python standard library:

```bash
python verify_release.py
```

To independently reaggregate the stored bootstrap statistics, use an isolated environment with the recorded analysis dependencies:

```bash
python -m pip install -r main_result/revision/requirements.txt
cd main_result
python validate_public_release.py
python run_reproduction.py --dry-run
```

The supplied validator reconstructs interval and frequency summaries from 8,000 aggregate procedure records: eight settings with 1,000 repetitions each. These rows describe bootstrap-level statistics, not individual respondents. It also checks that the historical 200 repetitions are the corresponding initial segment of the 1,000-repetition sequence. This verification does not refit models.

## Reproduce with authorized source data

Source microdata must be obtained from the KISDI AI Service User Policy Archive under the provider's access conditions. Place authorized workbooks at `main_result/private_raw/2023_data.xlsx`, `2024_data.xlsx`, and `2025_data.xlsx`. Source-audit stages also require the corresponding `YYYY_questionnaire.pdf` and `YYYY_codebook.xlsx`. Follow [main_result/README.md](main_result/README.md) for the complete execution stages and recorded environment. Additional boosting dependencies are in [requirements_round2.txt](main_result/revision/requirements_round2.txt).

The public execution entry point is `main_result/run_reproduction.py`. Supplementary Table S26 calls the earlier orchestration script `run_all_round2.py`; the distributed package uses `run_reproduction.py` instead. Use the distributed entry point and its `--stage` options. No missing script of the earlier name is required.

Run a fresh analysis in a separate working copy. Recalculation writes new results and execution receipts, so the original release hashes will then differ. Do not use Python `-O` or `-OO`, which disables analytical assertions.

The analysis figure stage writes analytical filenames into `main_result/JTAER_Figures/`. The top-level `JTAER_Figures/` contains the supplied, manuscript-numbered publication exports. Use `FIGURE_CROSSWALK.csv` to map analytical stems to publication numbers. Its `analytical_source_csv` column records locations in the analytical workspace; its `data_csv` column points to the released, numbered source data.

## Data scope

The distributed files contain code, metadata, and aggregate results. Respondent records, identifiers, row-level derived data, individual predictions, participant-specific resampling multiplicities, fold assignments, fitted models, and private caches are excluded. The ignore rules help keep locally regenerated private inputs and intermediates outside Git; uploads through other interfaces must follow the same scope. Provider microdata and questionnaire/codebook originals are not redistributed.

## Verification and provenance

For this publication, all 232 checksums in the two source manifests passed, all 46 Python files parsed, and all 36 JSON files loaded. The supplied aggregate validator passed with a maximum absolute reconstruction difference of approximately `9.93e-17`. The execution dry run listed 30 commands successfully. All 11 released PNGs exactly matched the embedded images in the attached 24 October 2026 manuscript and supplementary document. No model training, microdata reconstruction, or participant-level resampling was rerun for this upload.

The supplied `main_result/README.md` contains historical version-4/version-5 text. Its `PUBLIC_RELEASE_MANIFEST.json` retains the original pre-upload status and null repository commit. These are unmodified source-package records; this top-level README and `RELEASE_VALIDATION.json` describe the GitHub publication. Use the actual commit shown in this repository when citing a particular release.

The preceding repository remains available at [thinkpa81/mdpi](https://github.com/thinkpa81/mdpi). The manuscript identifies the original analysis by commit [`f3b0a46b86f8ab0a7e6093f17105d0f38d281824`](https://github.com/thinkpa81/mdpi/commit/f3b0a46b86f8ab0a7e6093f17105d0f38d281824). An earlier revision is archived at commit [`34f4172e3bebdb97fe5d5044dd51b237a7c82881`](https://github.com/thinkpa81/mdpi/tree/34f4172e3bebdb97fe5d5044dd51b237a7c82881/research_release_20261021). That revision predates the version-5 packages published here. The earlier repository was not changed by this upload.

No new software license or rights to redistribute provider data are granted by this packaging step.
