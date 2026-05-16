# Intro to AI HW3 Project

Final integrated branch: `chen`

This repository contains the integrated HW3 work for:

- Part 1 Task 1: within-subject EEG classification.
- Part 1 Task 2: cross-subject EEG classification.
- Part 2: dialogue continuity classification.

The `chen` branch is the stable integrated submission branch. The final task branches preserved for audit and reproducibility are:

- `chen-experiment/task1-final-integration`
- `chen-experiment/task2`
- `chen-experiment/part2-highscore`

Older experiment branches were cleaned from both local and remote during repository maintenance. Do not delete the preserved branches or worktrees unless a future maintenance task explicitly asks for that.

## Final Status Summary

| Part | Final method | Final candidate / anchor | Status |
| --- | --- | --- | --- |
| Part 1 Task 1 | Classical CSP, multiband, and ensemble EEG family | `task1_submission_ensemble_best.csv` and hedge `task1_submission_multiband_balanced.csv`; both public `0.62500` | Selected final within-subject solution. Later robust/model-ceiling work is diagnostic only. |
| Part 1 Task 2 | Cross-subject EEG diagnostics using LOSO validation with tangent/covariance/CSP-related variants | Historical public anchors `submission_task2_prior_public_adjusted.csv` and `submission_task2_prior_class2_less.csv`; both public `0.68750` | Preserved final Task 2 branch with robust generalization and pseudo-public/private validation analysis. |
| Part 2 | BERT/transformer common-split top-3 seed average ensemble | `part2/submission_bert_common_top3_seed_average.csv`; common validation Macro-F1 `0.9453`, threshold `0.48`, distribution `{0:217,1:283}` | Final recommended dialogue-continuity candidate. |

## Documentation Map

- `docs/515512_22_HW3_report.md`: assignment-style report with a final integration addendum.
- `docs/exp_results.md`: final integrated experiment summary and artifact policy.
- `part1/task1/task1_final_report.md`: final Task 1 decision record.
- `part1/task1/task1_stability_report.md`: Task 1 stability diagnostics.
- `part1/task1/task1_model_ceiling_report.md`: Task 1 model-ceiling diagnostics.
- `part1/task2/task2_pseudo_public_private_report.md`: Task 2 final diagnostics and pseudo-public/private validation.
- `part2/part2.ipynb`: Part 2 notebook required by the assignment.

## Artifact Policy

Generated data, submissions, outputs, model checkpoints, and caches remain ignored and should not be committed. This includes:

- `**/data/`
- `**/outputs/`
- `**/__pycache__/`
- `*.csv`
- model artifacts such as `*.pkl`, `*.pt`, `*.bin`, and `*.safetensors`

The final documentation intentionally records selected candidates and validation/public scores without committing generated submissions or raw data.
