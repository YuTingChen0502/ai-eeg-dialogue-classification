# Task 1 Model-Ceiling Experiment

This is an internal review artifact. It does not use test labels, infer public labels, or submit to Kaggle.

## Environment

- Worktree: `C:\Coding\Intro_to_AI\project1-task1`
- Branch: `chen-experiment/task1-model-ceiling`
- Data dir: `part1\task1\data`
- Shapes: x_train=(16, 45, 1125), y_train=(16,), x_test=(16, 45, 1125)
- Sampling rate: 250.0
- Class counts: `{"0": 4, "1": 4, "2": 4, "3": 4}`

## Reference Files

- Found: `task1_submission_ensemble_best.csv, task1_submission_multiband_balanced.csv, task1_submission_robust_csp.csv`
- Missing or invalid: `task1_submission_hybrid_two_public_winners.csv, task1_submission_multiband_best.csv`

## Methods

- Transductive OVR CSP: per-trial demeaning or common-average reference, bandpass filtering, compact one-vs-rest CSP, shrinkage LDA, and optional unlabeled-test joint channel z-scoring or Euclidean covariance alignment.
- Pairwise OVO CSP: six binary CSP plus shrinkage-LDA pair classifiers, combined by calibrated pair log-probability scores.
- Time-window robustness: full window plus 0.5-2.5s, 1.0-3.0s, and 1.5-3.5s motor-imagery windows; the window grid is intentionally small.
- Consensus-stable ensemble: model members selected by repeated CV, LOO F1, non-extreme prediction distribution, and avoidance of known failed patterns.

The unlabeled test set is used only inside label-free normalization/alignment transforms. During validation, the held-out fold is treated analogously as unlabeled evaluation data for those transforms.

## Search Scope

- Configs evaluated: 19
- Repeated CV: Stratified 4-fold x 3 repeats
- LOO: aggregate predictions across all 16 held-out trials
- Required bands covered: `alpha_mu_8_13, beta_13_30, broad_4_40, high_gamma_70_125`

## Top Internal Configs

| rank | family | feature_set | preprocess | adaptation | window | score | rskf3_f1 | rskf3_std | loo_f1 | dist | risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ovr_csp | alpha_mu_8_13 | car | none | full | 0.7085 | 0.6111 | 0.2390 | 0.6762 | {"0": 2, "1": 4, "2": 4, "3": 6} | near_failed_robust_csp |
| 2 | ovr_csp | alpha_mu_8_13 | car | joint_zscore | full | 0.7069 | 0.6250 | 0.2269 | 0.6000 | {"0": 2, "1": 4, "2": 4, "3": 6} | near_failed_robust_csp |
| 3 | ovr_csp | alpha_mu_8_13 | car | ea_joint | full | 0.7008 | 0.5833 | 0.2846 | 0.8115 | {"0": 4, "1": 2, "2": 2, "3": 8} | extreme_distribution,high_cv_variance |
| 4 | ovo_csp | alpha_mu_8_13 | car | none | full | 0.5215 | 0.4340 | 0.2291 | 0.6250 | {"0": 4, "1": 3, "2": 3, "3": 6} | none |
| 5 | ovr_csp | fb_required_all | car | ea_joint | full | 0.7684 | 0.6667 | 0.2722 | 0.7333 | {"0": 2, "1": 4, "2": 0, "3": 10} | class_collapse,extreme_distribution,high_cv_variance |
| 6 | ovr_csp | alpha_mu_8_13 | car | ea_joint | 1.0-3.0s | 0.6147 | 0.5347 | 0.1411 | 0.4893 | {"0": 1, "1": 7, "2": 7, "3": 1} | feet_heavy |
| 7 | ovr_csp | beta_13_30 | car | ea_joint | full | 0.4706 | 0.4618 | 0.2329 | 0.3145 | {"0": 1, "1": 4, "2": 2, "3": 9} | extreme_distribution,weak_loo |
| 8 | ovr_csp | high_gamma_70_125 | car | ea_joint | full | 0.4988 | 0.4479 | 0.1940 | 0.4365 | {"0": 8, "1": 6, "2": 1, "3": 1} | extreme_distribution,weak_loo,large_shift_vs_ensemble_best |
| 9 | ovr_csp | fb_required_all | car | none | full | 0.5739 | 0.4722 | 0.2792 | 0.7417 | {"0": 3, "1": 3, "2": 0, "3": 10} | class_collapse,extreme_distribution,high_cv_variance |
| 10 | ovr_csp | alpha_mu_8_13 | car | ea_joint | 0.5-2.5s | 0.3500 | 0.3847 | 0.2928 | 0.2123 | {"0": 6, "1": 2, "2": 2, "3": 6} | high_cv_variance,weak_loo,large_shift_vs_ensemble_best |
| 11 | ovr_csp | broad_4_40 | car | none | full | 0.3640 | 0.3194 | 0.1979 | 0.4159 | {"0": 4, "1": 2, "2": 2, "3": 8} | extreme_distribution,weak_loo |
| 12 | ovo_csp | alpha_mu_8_13 | car | ea_joint | full | 0.5297 | 0.4896 | 0.2283 | 0.4345 | {"0": 6, "1": 0, "2": 1, "3": 9} | class_collapse,extreme_distribution,weak_loo |

## Candidate Results

| candidate | file | members | dist | rskf3_f1 | rskf3_std | loo_f1 | class_f1 | hamming | changed_rows | risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transductive | task1_submission_model_ceiling_transductive.csv | 1 | {"0": 2, "1": 4, "2": 4, "3": 6} | 0.6250 | 0.2269 | 0.6000 | {"0": 0.8, "1": 0.6666666666666666, "2": 0.3333333333333333, "3": 0.6} | {"task1_submission_ensemble_best.csv": 9, "task1_submission_multiband_balanced.csv": 10, "task1_submission_robust_csp.csv": 1} | {"task1_submission_ensemble_best.csv": [0, 3, 6, 8, 9, 10, 12, 13, 15], "task1_submission_multiband_balanced.csv": [0, 1, 2, 3, 5, 10, 11, 13, 14, 15], "task1_submission_robust_csp.csv": [12]} | near_failed_robust_csp |
| ovo_csp | task1_submission_model_ceiling_ovo_csp.csv | 1 | {"0": 4, "1": 3, "2": 3, "3": 6} | 0.4340 | 0.2291 | 0.6250 | {"0": 0.5, "1": 0.5, "2": 0.75, "3": 0.75} | {"task1_submission_ensemble_best.csv": 8, "task1_submission_multiband_balanced.csv": 11, "task1_submission_robust_csp.csv": 6} | {"task1_submission_ensemble_best.csv": [1, 3, 9, 10, 12, 13, 14, 15], "task1_submission_multiband_balanced.csv": [1, 2, 3, 5, 6, 8, 10, 11, 13, 14, 15], "task1_submission_robust_csp.csv": [0, 1, 6, 8, 12, 14]} | none |
| consensus | task1_submission_model_ceiling_consensus.csv | 3 | {"0": 5, "1": 2, "2": 2, "3": 7} | 0.5556 | 0.1843 | 0.6083 | {"0": 0.6, "1": 0.3333333333333333, "2": 0.75, "3": 0.75} | {"task1_submission_ensemble_best.csv": 8, "task1_submission_multiband_balanced.csv": 13, "task1_submission_robust_csp.csv": 6} | {"task1_submission_ensemble_best.csv": [0, 1, 3, 10, 12, 13, 14, 15], "task1_submission_multiband_balanced.csv": [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 15], "task1_submission_robust_csp.csv": [1, 6, 8, 9, 12, 14]} | none |

## Recommendation

No new candidate is worth uploading from this phase. The current preferred hedge remains `task1_submission_ensemble_best.csv` plus `task1_submission_multiband_balanced.csv`.

Generated CSVs and diagnostics are review-only artifacts and should remain uncommitted.
