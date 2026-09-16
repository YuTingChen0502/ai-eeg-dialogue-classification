# EEG & Dialogue Classification

Team project covering two machine-learning tasks: cross-subject EEG motor-execution classification and dialogue-continuity classification.

The repository contains experiment code, training/inference pipelines, notebooks, and the final submission reports for both tasks.

## Task 1 — Cross-subject EEG classification

The team pipeline focused on robustness across unseen subjects rather than within-subject accuracy.

Final approach:

- 8–30 Hz EEG preprocessing with notch filtering and trial cropping
- per-subject Euclidean Alignment to reduce subject-specific covariance shift
- sliding-window covariance features projected to Riemannian tangent space
- Logistic Regression as the primary classifier
- confidence-gated EEGNet ensemble for low-confidence trials

Selected result:

- Kaggle public leaderboard: 0.8750

A key observation was that cross-subject distribution shift dominated model choice: Euclidean Alignment produced a much larger gain than simply increasing model complexity.

## Task 2 — Dialogue continuity classification

The project compared classical text models with several transformer-based approaches.

Evaluated methods included:

- TF-IDF + LinearSVC
- DistilBERT
- DeBERTa-v3-base
- RoBERTa-base
- OOF stacking and probabilistic ensembling
- validation-based threshold tuning

Final selected model:

| Metric | Result |
|---|---:|
| OOF Macro-F1 | 0.968701 |
| Kaggle public score | 0.9686 |
| Threshold | 0.510 |

The final submission used RoBERTa-base after it outperformed the classical baseline and the more complex stacking/ensemble alternatives on out-of-fold validation.

## My contributions

My work was concentrated on experiment design, model comparison, and final integration, particularly for Part 2.

- built and evaluated Part 2 candidate experiment workflows;
- explored transformer and ensemble variants, including common-split diagnostics and stacking;
- implemented out-of-fold validation and threshold-selection workflows;
- integrated the final Part 2 model-selection and reproducibility path;
- helped finalize Task 1 integration and the final submission/report package.

The repository also contains work from other team members; the project is presented here as a team project rather than as individual authorship of every component.

## Repository structure

```text
part1/       EEG experiments and inference pipelines
part2/       dialogue-classification experiments and transformer workflows
SUBMISSION/  final code/report packaging
docs/        supporting project material
```

Detailed experimental tables and methodology are preserved in `SUBMISSION/splited_report/`.

## Notes

Leaderboard values above are public leaderboard scores only and should not be interpreted as private-test or generalization guarantees.
