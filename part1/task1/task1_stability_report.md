# Task 1 Internal Stability Analysis

This is an internal-only diagnostic. It does not create Kaggle candidate CSVs and does not use
public row flipping.

## Setup

- Models: regularized one-vs-rest CSP + shrinkage LDA
- Preprocessing variants: per-channel demean, common-average reference
- CSP components: 2, 4
- CSP covariance shrinkage: 0.15, 0.25, 0.35
- Validation: repeated stratified 4-fold CV with 20 repeats, plus leave-one-out
- Required bands covered individually:
  - 8-13 Hz
  - 13-30 Hz
  - 4-40 Hz
  - 70-125 Hz

## Band Summary

| Band | Top selection score | Mean selection score | Top repeated-CV macro-F1 | Top LOO macro-F1 |
|---|---:|---:|---:|---:|
| 8-13 Hz | 0.6579 | 0.5386 | 0.5958 | 0.6000 |
| 70-125 Hz | 0.5775 | 0.3626 | 0.4921 | 0.7560 |
| 13-30 Hz | 0.4902 | 0.3838 | 0.4625 | 0.4236 |
| 4-40 Hz | 0.4226 | 0.3624 | 0.4089 | 0.4921 |

Interpretation: 8-13 Hz is the most stable band by repeated-CV selection score. 70-125 Hz has
one strong LOO result but weaker repeated-CV stability, so it remains high-variance evidence.

## Consensus Pool

The consensus pool contains 14 low-degree-of-freedom CSP+LDA models. It includes the top two
configs from each required band, then fills with top-scoring configs while preserving required
band coverage.

Consensus test distribution:

- label 0: 1
- label 1: 8
- label 2: 4
- label 3: 3

Consensus-stable rows:

- stable: `[2, 4, 5, 11, 12]`
- mixed: `[3, 6, 7, 9, 10, 14, 15]`
- unstable: `[0, 1, 8, 13]`

## Reference Comparison

References found:

- `task1_submission_ensemble_best.csv`
- `task1_submission_multiband_balanced.csv`

Reference missing:

- `task1_submission_hybrid_two_public_winners.csv`

Hamming distance from internal consensus:

| Reference | Hamming | Changed row ids |
|---|---:|---|
| `task1_submission_ensemble_best.csv` | 6 | `[7, 8, 9, 12, 13, 15]` |
| `task1_submission_multiband_balanced.csv` | 13 | `[1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]` |

## Recommendation

This analysis does not overturn the current 0.625 hedge. The internal consensus still has
meaningful uncertainty on several rows and remains much closer to `ensemble_best` than to
`multiband_balanced`. Because `robust_csp` already failed as a public diagnostic, no additional
Kaggle probing is justified from this evidence.

Current preferred Task 1 hedge remains:

1. `task1_submission_ensemble_best.csv`
2. `task1_submission_multiband_balanced.csv`

## Exact Next Step

Use this analysis for the report: emphasize that alpha/mu CSP is the most stable internal model
family, but public transfer is unreliable with only 16 training trials. If continuing model-side
work, the next non-public step is to inspect the unstable rows `[0, 1, 8, 13]` with time-window
robustness analysis inside the same CSP+LDA framework.
