# JTAER minor revision focused raw-data diagnostics

These diagnostics address the generative-AI experience item and participant overlap within the hesitation groups. They leave the original fitted models, original comparison families, and published loss estimates unchanged.

## Inputs

Place the authorized KISDI workbooks in an input directory using the aliases `panel_2023.xlsx`, `panel_2024.xlsx`, and `panel_2025.xlsx`. The source is the `Numeric` sheet in each workbook. The script verifies that each workbook's SHA-256 matches the source manifest of revision commit `af06ecda65c0f5768cbe2b3449bb46ebf229edca`.

| Alias | Original source name | SHA-256 |
|---|---|---|
| panel_2023.xlsx | 2023_2023년 지능정보사회 이용자 패널조사_data(종단면가중치 포함).xlsx | c38782a595dc9ee898cb64a3ebbdde40099c09cca655454eabc7b02316efb0a9 |
| panel_2024.xlsx | 2024_2024년 지능정보사회 이용자 패널조사_data(종단면가중치 포함).xlsx | 65fa040101d784b2225eafea9a7a8e5660c5fca9f1b962e342cd81b9cb689f5c |
| panel_2025.xlsx | 2025_2025년 지능정보사회 이용자 패널조사_data(종단면가중치 포함).xlsx | 16867d676338e22da82a84c72b667fe5764ff266a03d6e8086182de138a2c185 |

Source microdata are not part of this reproducibility supplement. Obtain them through the provider under the applicable access conditions.

## Execution

```bash
python minor_raw_diagnostics.py --raw-dir /path/to/authorized_inputs --output-dir /path/to/aggregate_outputs
```

Dependencies used for this run: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0, and an Excel-reading engine compatible with pandas. The analysis writes aggregate CSV/JSON outputs only; it does not write participant records.

## Definitions

- A is reported generative-AI use experience, with no specified recall window. Source variables are `q23` in 2023, `Q24` in 2024, and `q20` in 2025. Source response code 1 denotes experience and code 2 denotes no experience. The 2025 A value is audited but is not used as a predictor in either original transition.
- The outcome is next-year reported any use of consumer-domain intelligent services: source response codes 1–4 are use; 5 and 9 are no past-year use. Code 9 is a valid response.
- Baseline hesitation codes 1–2 define the restricted cohorts. Full linked cohorts retain all baseline hesitation categories.
- Overlap in the full-panel group comparison is membership in the full linked development sample of 3,895 people. It is distinct from overlap with the restricted development sample of 1,565 people.

## Output interpretation

`A_prevalence.csv` reports A prevalence in the 1,565-person development cohort and 1,520-person evaluation cohort. The two cohorts contain 641 of the same participants. Its Wilson intervals are pointwise descriptions within each cohort.

`A_outcome_association.csv` and `A_outcome_2x2_counts.csv` report the unweighted association of baseline A with the next-year outcome. The odds ratio uses a pointwise log-Wald interval; the risk difference uses the Newcombe difference of Wilson intervals. These unadjusted associations do not estimate the conditional Shapley allocation or the incremental performance of adding A to a fitted model.

`full_panel_group_overlap.csv` gives overlap counts and proportions for the hesitant and other groups and for the two components of the other group. The restricted-development overlap columns are included only to distinguish the different reference training samples.

`A_shared_participant_transitions.csv` is an optional descriptive diagnostic among the same 641 participants. It separates an observed increase within these participants from the change in aggregate cohort composition.

`minor_diagnostics_manifest.json` contains source hashes, sample controls, software versions, and the period-specific aggregate estimates. The supplementary paired counts describe changes among the same participants without introducing an additional inferential comparison.

The separate `a_item_wording_audit.json` records the audit of the original questionnaires and codebooks. Core wording and response codes are stable; accompanying examples and survey positions change. This audit does not establish measurement invariance.

All original sample sizes, event counts, and overlap controls passed before these outputs were written. The diagnostic results are based on the authorized source records; no generated or synthetic observations were used.
