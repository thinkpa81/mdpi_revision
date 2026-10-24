# JTAER final research materials

**Questionnaire Item Budgets and Incremental Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Temporal Evidence from a Korean Panel**

Soonho Jang and Eung-Kyo Suh, Dankook University.

This repository contains the analysis code, aggregate results, provenance, and figure data supporting the 24 October 2026 manuscript and supplementary materials. The latest editorial revision clarifies the retrospective analysis scope, measurement diagnostics, prospective validation needs, and reproducibility information across seven submission documents. All 34 reproduced editor/reviewer comments remain unchanged. All 105 detailed response locators were verified against the revised documents, with 23 page/line locations updated. Clean and marked versions have matching text, equations, and page counts. Current document identities and verification scope are recorded in [EDITORIAL_ALIGNMENT_CHECK_20261024.json](EDITORIAL_ALIGNMENT_CHECK_20261024.json). The preceding response audit is retained in [RESPONSE_ALIGNMENT_CHECK_20261024.json](RESPONSE_ALIGNMENT_CHECK_20261024.json); earlier title/release checks remain in [FINAL_ALIGNMENT_CHECK.json](FINAL_ALIGNMENT_CHECK.json). Submission documents themselves are not included.

[한국어 안내](README_KO.md)

## Contents

| Path | Purpose |
| --- | --- |
| [main_result/revision/code](main_result/revision/code) | Current reconstruction, model, inference, bootstrap, sensitivity, diagnostic, and plotting code |
| [main_result/revision/tables](main_result/revision/tables) | Aggregate model results and bootstrap/holdout summaries |
| [main_result/revision/results](main_result/revision/results) | Interval, selection, overlap, regional, and adjacent-service diagnostics |
| [main_result/revision/audit](main_result/revision/audit) | Source hashes, protocols, recorded versions, and execution receipts |
| [main_result/original_results](main_result/original_results) | Original code and aggregate evidence required for harmonization, numerical comparisons, retained nested results, and Supplementary Table S25 |
| [JTAER_Figures](JTAER_Figures) | Main Figures 1–7 and Supplementary Figures S1–S4, aggregate plot data, and the final figure crosswalk |
| [SOURCE_RECHECK_20261024.json](SOURCE_RECHECK_20261024.json) | Source reconstruction and adjacent-service recheck cited in Supplementary Section S10 |
| [FINAL_ALIGNMENT_CHECK.json](FINAL_ALIGNMENT_CHECK.json) | Earlier title/release alignment and archived removals |
| [RESPONSE_ALIGNMENT_CHECK_20261024.json](RESPONSE_ALIGNMENT_CHECK_20261024.json) | Preceding response/document alignment and aggregate-check scope |
| [EDITORIAL_ALIGNMENT_CHECK_20261024.json](EDITORIAL_ALIGNMENT_CHECK_20261024.json) | Current document identities, editorial alignment, and locator/figure checks |
| [FILE_MANIFEST_SHA256.json](FILE_MANIFEST_SHA256.json) | Current repository file hashes |

Supplementary Table S26 maps the analysis groups to these scripts and outputs. Existing relative paths are preserved so that the documented pipeline continues to resolve its inputs. The earlier 200-repetition bootstrap records remain because the 1,000-repetition analysis and public validator compare against them. Their role is provenance and numerical verification, rather than an additional independent sample.

## Verify the public materials

```bash
python verify_release.py
python -m pip install -r main_result/revision/requirements.txt
cd main_result
python validate_public_release.py
python run_reproduction.py --dry-run
```

The integrity check uses only the Python standard library. The numerical validator reaggregates the 8,000 procedure-level bootstrap records, checks the initial 200-repetition segment, and reconstructs interval and selection-frequency summaries. These records summarize analysis repetitions; they contain no respondent rows. This verification does not refit models. The preceding response audit reaggregated the 8,000 records and checked 98 downloaded evidence files against the base manifest; it did not rerun the complete repository validator or access source microdata. Its maximum absolute reaggregation difference was below 9.94e-17. The current editorial step verifies document consistency and updates metadata; it does not repeat that reaggregation or refit models. Scientific code, results, and figures are unchanged.

## Reproduce with authorized source data

Obtain the 2023–2025 workbooks from the KISDI AI Service User Policy Archive under the provider's access conditions. Place the authorized `YYYY_data.xlsx`, `YYYY_codebook.xlsx`, and `YYYY_questionnaire.pdf` files in `main_result/private_raw/`. Follow [main_result/README.md](main_result/README.md) for dependencies, scope, and execution order. Additional boosting dependencies are listed in `main_result/revision/requirements_round2.txt`.

The execution entry point is `main_result/run_reproduction.py`, written as `../../run_reproduction.py` relative to `main_result/revision/code/` in Supplementary Table S26. Run the stages in a separate working copy; fresh execution replaces results and execution receipts. Run Python without `-O` or `-OO` so the analytical assertions remain active.

The analysis figure stage writes to `main_result/JTAER_Figures/`. The checked publication exports are in the top-level [JTAER_Figures](JTAER_Figures) directory. Its [FIGURE_CROSSWALK.csv](JTAER_Figures/FIGURE_CROSSWALK.csv) maps analytical filenames to publication numbers and available aggregate plot data. All 11 figure captions and embedded PNGs were checked against the revised documents. Captions, image bytes, and plot data remain unchanged; the figure manifest identifies the current clean documents.

## Public-data boundary

Published materials consist of code, metadata, aggregate statistics, and figures. Respondent records, identifiers, individual predictions, participant-specific resampling multiplicities, fold assignments, fitted objects, and private caches are excluded. The original provider workbooks, questionnaires, and codebooks must be obtained separately. Manuscripts, supplementary-document files, response letters, correspondence, and source ZIP copies are excluded. Ignore rules reinforce this boundary for local Git use; the same scope applies to uploads through other interfaces.

## Source recheck and analysis scope

The separate 24 October source recheck confirmed the hashes of nine authorized workbooks, codebooks, and questionnaires. Reruns of `source_audit.py`, `codebook_audit.py`, `questionnaire_audit.py`, and `adjacent_service_diagnostics.py` reproduced four private analysis-dataset hashes and 20 released aggregate CSVs exactly. The adjacent-service check repeated 10,000 participant draws per cohort, 36 contrasts, and 24 conditional tests. Details are in [SOURCE_RECHECK_20261024.json](SOURCE_RECHECK_20261024.json).

That check did not refit all predictive models or repeat the 1,000-repetition development bootstrap. This final repository alignment also preserves existing scientific results. The full-process bootstrap covers budgets 1 and 4 with the original ridge/HGB grids and the restricted whole evaluation cohort. Other result files retain their stated inference scopes. The study is a retrospective, exploratory secondary analysis: later-period outcomes were already examined, additional tuning is confined to development data, and reuse of the evaluation cohort provides further exploratory assessment. Original comparison families and additional sensitivity analyses retain their stated scopes. Table S28 distinguishes observed instrument correspondence and response-change diagnostics from a proposed concurrent wording study, which was not implemented. These diagnostics do not identify formal longitudinal measurement equivalence or a causal wording effect. Field evaluation of collection policies requires stable measures and directly measured acquisition costs.

## Archived provenance and citation

The final documents cite commit [`af06ecda65c0f5768cbe2b3449bb46ebf229edca`](https://github.com/thinkpa81/mdpi_revision/commit/af06ecda65c0f5768cbe2b3449bb46ebf229edca). That immutable snapshot remains available and is the numerical source of this alignment. The current tree removes eight unused earlier implementation copies, one redundant editorial caption file, and one superseded upload-preparation receipt. Each removed file and its archived URL are recorded in [FINAL_ALIGNMENT_CHECK.json](FINAL_ALIGNMENT_CHECK.json). Required historical aggregate results and scientific execution receipts remain in the current tree.

The original analysis is available in [thinkpa81/mdpi](https://github.com/thinkpa81/mdpi) at commit `f3b0a46b86f8ab0a7e6093f17105d0f38d281824`; the preceding revision is under `research_release_20261021/` at commit `34f4172e3bebdb97fe5d5044dd51b237a7c82881`. The earlier repository is unchanged. Cite the actual GitHub commit for the snapshot used. No additional software license or provider-data redistribution rights are granted by this alignment.

