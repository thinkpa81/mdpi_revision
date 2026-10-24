# Budget-constrained questionnaire collection protocol

This plan preceded its new calculations, after the evaluation data had already been examined in the wider project. It describes a retrospective reanalysis, not preregistration or an untouched evaluation sample.

- Data: the 2023-to-2024 development sample (1,565 respondents) and 2024-to-2025 evaluation sample (1,520 respondents), retaining the original input definitions.
- Information availability: the history-available condition admits all 256 portfolios and P0. The history-unavailable condition admits the 128 portfolios without H. Availability of H does not require the selected portfolio to use H.
- Collection budgets: every integer from 0 through 30. Already-held H has zero additional acquisition cost. Other costs are original questionnaire item counts, not measured response time, privacy harm, or legal sensitivity.
- Learners: ridge and HGB are analyzed separately. Ridge C is in {0.001, 0.01, 0.1, 1, 10}; HGB retains the original two configurations.
- Selection: minimize development out-of-fold log loss among portfolios within budget. Break exact ties by actual additional item count and then portfolio name. No preservation margin enters selection.
- Full-development policy: validate and reuse the original out-of-fold predictions and tuning results, and compute P0 on the identical five folds. The development-selection function does not read evaluation data.
- Nested evaluation: five outer stratified folds with two repetitions; five inner stratified folds within each outer training set. All candidates share splits. Preprocessing, tuning, and portfolio selection remain within training data. Refit only the selected budget policies for outer-fold assessment.
- Nested summary: average the two repetition-specific participant-level out-of-fold losses. The two repetitions do not increase the independent respondent count. Fold selection frequencies are descriptive.
- Temporal evaluation: freeze development policies before assessing the later sample. Verify prediction caches against input bytes, configurations, identifier ordering, and refits. Evaluation performance does not retrospectively change the development selection rule.
- Distribution: publish aggregate CSVs, configurations, and code. Keep respondent identifiers, individual predictions, and raw records in private directories.
- Reporting: retain deterioration and benchmark underperformance. Do not alter budgets, selection rules, or split counts during output generation to favor the results.
