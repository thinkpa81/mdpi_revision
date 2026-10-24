# JTAER submission figure captions

Numbering and captions match the revised main manuscript and supplementary materials.

## Figure 1. Study design and analysis workflow.

The original development and temporal cohorts are reconstructed from the 2023–2025 KISDI panel. Development-only selection precedes later-period scoring; participant-union resampling repeats the complete focal-budget procedure. Additional analyses examine calibration, prediction population, expanded learner searches including XGBoost, LightGBM and CatBoost, regional recoding, and adjacent-service response changes. The temporal period was already examined in the submitted study.

## Figure 2. Development and temporal performance across item budgets.

The development curve is the mean of two retained nested cross-validation repetitions. Its shaded range spans the two repetition-specific means and is descriptive. Temporal bands are pointwise 95% participant-bootstrap intervals for fixed predictions (5,000 draws; seed 2026102002); they do not include tuning or selection uncertainty and do not provide joint coverage over budgets. Lower log loss is better. The dashed reference identifies temporal P0 or NONE. See source CSV for all 31 budgets.

## Figure 3. Temporal loss differences of focal original selections.

Positive Δ indicates higher loss than P0 (H available) or NONE (H unavailable). Bars are two-sided simultaneous 95% maximum-absolute centered-bootstrap intervals, separately within each original algorithm × history-availability family containing all selected budgets and both references; 5,000 paired evaluation resamples, n=1,520. The eight displayed contrasts are not a newly restricted pooled family. Predictions are fixed.

## Figure 4. Selection recurrence and uncertainty after repeating model development.

Left: frequency of the original portfolio across 1,000 participant-union bootstrap repetitions. Right: original point estimate and pointwise 95% percentile interval for the complete procedure against P0 or NONE. Shared participants receive common multiplicities across periods; duplicate copies remain together in inner folds. Selection, original-grid tuning, preprocessing, and fitting are repeated for budgets 1 and 4. These exploratory intervals are not multiplicity-adjusted and have finite Monte Carlo precision. The displayed intervals use the primary percentile method; Table S12 reports basic-interval sensitivity and bootstrap bias.

## Figure 5. Within-period and later-period performance at matched training size.

Twenty stratified 70/30 development splits repeat inner five-fold tuning and budget-4 selection. Each trained model is applied to its 470-person within-period holdout and the later 1,520-person cohort. Points are mean model-minus-benchmark loss differences; bars are 2.5th–97.5th split percentiles. They summarize overlapping-split variability, not confidence intervals. Wording, time, and sample composition change together.

## Figure 6. Additional-information value varies by model and prediction population.

Four algorithms use development-only expanded searches. Each panel contains a separate eight-contrast family: full with H versus P0 and full without H versus NONE, across four learners. Bars are two-sided simultaneous 95% maximum-absolute centered paired-bootstrap intervals within each panel (5,000 repetitions), conditional on fitted models. Development/evaluation n=1,565/1,520 for the restricted population and 3,895/3,741 for all linked respondents. These are retrospective sensitivity comparisons.

## Figure 7. Block contributions across development and temporal evaluation.

Exact seven-block loss-reduction allocations per original item for the two original learners and both history settings. Positive values indicate improved prediction. All 84 cells share one color scale. The 28 development cells reuse tuning-fold out-of-fold predictions and are descriptive. The 56 temporal cells use fixed fitted predictions. Outlined cells mark temporal allocations whose 56-contrast simultaneous 95% intervals exclude zero; all seven indicate higher loss. No positive allocation has an interval excluding zero. Full interval estimates appear in Table S21 and person-disjoint Figure S4. Development, whole temporal, and person-disjoint sample sizes are 1,565, 1,520, and 879. H is fixed present or absent. Per-item allocation is an efficiency diagnostic, not a causal effect or a cost-weighted Shapley game.

## Figure S1. Absolute losses of original full models and no-additional-collection benchmarks.

Original configurations were refitted from the source workbooks. Temporal evaluation n=1,520. Bars are pointwise 95% participant-bootstrap intervals conditional on fitted predictions (2,000 repetitions). Paired contrast tests provide the inferential comparisons between models. H denotes prior self-reported use.

## Figure S2. Affine calibration and ranking diagnostics.

Raw log loss equals empirical evaluation outcome entropy minus affine resolution plus affine miscalibration. Components are relative to apparent intercept-and-slope logistic recalibration, not a nonparametric decomposition. Bars are pointwise 95% percentile intervals from 500 resamples that refit the recalibrator while keeping base predictions fixed. Cross-fitted recalibration in Table S14 limits reliance on optimistic apparent gains. All diagnostics use evaluation outcomes.

## Figure S3. Hesitation-related predictive heterogeneity under common full-panel-trained models.

Hesitant respondents (n=1,520; codes 1–2) are compared with other linked respondents (n=2,221; codes 3, 4, and 9). Positive contrasts indicate less benefit or greater harm from full information in the hesitant group. Bars are two-sided simultaneous 95% intervals across eight full-model contrasts, using 5,000 bootstrap resamples independently within each group. This is conditional predictive heterogeneity, not a causal moderator effect.

## Figure S4. Block contributions in person-disjoint evaluation.

Exact seven-block loss-reduction allocations in the 879-person subset, in total and per-original-item units. Positive values indicate improved prediction. H is fixed present or absent. Bars use the same simultaneous 56-contrast family covering both evaluation domains, algorithms, history settings, and blocks (5,000 fixed-prediction resamples). Total and per-item panels use separate common scales; main Figure 7 compares development, whole temporal, and person-disjoint per-item values.
