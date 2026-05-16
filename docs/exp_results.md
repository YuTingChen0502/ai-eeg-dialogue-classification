# HW3 Final Integrated Experiment Summary

This file supersedes older upload-priority notes. The final integrated branch is `chen`, and the preserved task branches are:

- `chen-experiment/task1-final-integration`
- `chen-experiment/task2`
- `chen-experiment/part2-highscore`

Older experiment branches were cleaned from local and remote. Generated data, submissions, outputs, checkpoints, and caches remain ignored and should not be committed.

## Final Results

| Part | Final branch | Method family | Selected candidate / anchor | Key evidence | Final decision |
| --- | --- | --- | --- | --- | --- |
| Part 1 Task 1 | `chen-experiment/task1-final-integration` | CSP, multiband CSP, and small classical ensembles | `task1_submission_ensemble_best.csv` with distribution `{0:1,1:7,2:1,3:7}`; hedge `task1_submission_multiband_balanced.csv` with distribution `{0:6,1:2,2:5,3:3}` | Both reached public `0.62500`; required bands `8-13`, `13-30`, `4-40`, and `70-125` Hz were compared; robust/model-ceiling phases did not justify replacement | Keep the earlier CSP/multiband/ensemble family as final. |
| Part 1 Task 2 | `chen-experiment/task2` | Cross-subject EEG generalization diagnostics using LOSO, tangent/covariance, and CSP-related variants | Historical public anchors `submission_task2_prior_public_adjusted.csv` `{0:6,1:8,2:9,3:9}` and `submission_task2_prior_class2_less.csv` `{0:6,1:9,2:8,3:9}` | Both anchors reached public `0.68750`; diagnostics show class 2 is the weakest class and cross-subject shift is the main risk | Preserve final Task 2 branch and document diagnostics; public score is only a sparse guardrail. |
| Part 2 | `chen-experiment/part2-highscore` | BERT/transformer common-split top-3 seed average ensemble | `part2/submission_bert_common_top3_seed_average.csv` | Seeds `42/13/7`; common validation Macro-F1 `0.9453`; threshold `0.48`; distribution `{0:217,1:283}`; Hamming distance 7 vs seed 42 on IDs `1278, 419, 1974, 1949, 1694, 479, 549`; CSV validated | Final recommended dialogue-continuity candidate. |

## Rejected Or Diagnostic Candidates

Task 1 rejected candidates:

- `task1_submission_multiband_best.csv`: distribution `{0:2,1:1,2:10,3:3}`, public `0.25000`; rejected as feet-heavy overfit.
- `task1_submission_robust_csp.csv`: distribution `{0:2,1:4,2:3,3:7}`, public `0.50000`; rejected as a failed diagnostic.
- Model-ceiling candidates were not submitted because none clearly improved the final hedge.

Task 2 diagnostics:

- Pseudo-public/private experiments used labels only inside held-out training-subject simulation.
- No inferred real-test labels were used as training data.
- No legal pseudo-public/private method cleared the configured validation threshold strongly enough to replace the historical public anchors.

Part 2 diagnostics:

- Classical TF-IDF and rule/endpoint features were useful baselines.
- The final BERT common-split top-3 seed average was preferred over single-seed and classical candidates.

## Guardrails

- No external data.
- No pretrained models for Part 1.
- No test labels.
- No public row-flipping loop.
- No new model experiments during final documentation maintenance.
- Generated CSVs, raw data, caches, checkpoints, and outputs stay ignored/uncommitted.
