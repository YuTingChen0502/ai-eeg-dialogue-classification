# Task 1 Final Report

## Task Overview

Task 1 is a within-subject EEG classification problem. The goal is to predict four motor-imagery/rest classes from each trial:

- 0: left hand
- 1: right hand
- 2: feet
- 3: rest

The labeled training set is extremely small: `part1/task1/data/train.npz` contains `x_train=(16, 45, 1125)` and `y_train=(16,)`. The unlabeled test set at `part1/task1/data/test.npz` contains `x_test=(16, 45, 1125)`. Because there are only 16 labeled trials and exactly four examples per class, every local validation split has high variance. A single changed trial can noticeably move macro-F1, so local validation was treated as a diagnostic signal instead of a reliable estimate of private leaderboard performance.

The required submission format is a CSV with header `id,label`, exactly 16 prediction rows, preserved test order, and labels restricted to `{0, 1, 2, 3}`.

## Preprocessing

The final preferred solution family used classical EEG preprocessing and low-degree-of-freedom classifiers:

- Per-channel demeaning and normalization to reduce trial-level offset and scale differences.
- Bandpass filtering for the assignment-required bands and multiband/filterbank variants.
- CSP spatial filters with log-variance features.
- Common-average reference was included in the robust/model-ceiling diagnostic phases as an optional preprocessing variant.
- No external data was used.
- No pretrained models were used.

## Required Band Comparison

The stability phase evaluated regularized one-vs-rest CSP plus shrinkage LDA with repeated stratified CV and leave-one-out validation. The most useful finding was not a single absolute score, but the relative stability of each required band.

| Band | Repeated CV macro-F1 | LOO macro-F1 | Comment |
| --- | ---: | ---: | --- |
| 8-13 Hz | 0.5958 | 0.6000 | Most stable band. This alpha/mu range gave the best repeated-CV evidence. |
| 13-30 Hz | 0.4625 | 0.4236 | Weaker and less consistent than 8-13 Hz. |
| 4-40 Hz | 0.4089 | 0.4921 | Broad coverage, but noisy on this tiny labeled set. |
| 70-125 Hz | 0.4921 | 0.7560 | Promising in LOO, but unstable under repeated CV, so it was treated cautiously. |

## Methods Tried

The early solution family used CSP, multiband CSP, and small ensembles. This produced the best public hedge and remained the final selected approach.

The robust diagnostic phase expanded the search around regularized CSP and shrinkage/classical classifiers. It compared per-channel demeaning, common-average reference, required single bands, filterbank combinations, and repeated validation protocols. It also tested whether candidate predictions stayed close to already known public-safe files.

The stability analysis focused on whether stable CSP plus shrinkage-LDA models agreed on the unlabeled test rows. It confirmed that the 8-13 Hz band was the most stable internal signal, but also showed several unstable rows and did not justify additional public probing.

The model-ceiling phase tested more aggressive but still label-free variants:

- Transductive OVR CSP using train labels plus unlabeled test signals for normalization/alignment only.
- Joint train/test z-score normalization.
- Euclidean covariance alignment.
- Pairwise OVO CSP.
- Time-window robustness across full window, 0.5-2.5s, 1.0-3.0s, and 1.5-3.5s windows.
- A consensus ensemble selected from internally stable members.

These later experiments were useful diagnostics, but none produced enough evidence to replace the earlier 0.625 public hedge.

## Results

| Candidate | Prediction distribution | Local validation | Public score | Final decision |
| --- | --- | --- | ---: | --- |
| `ensemble_best` | `{0: 1, 1: 7, 2: 1, 3: 7}` | Earlier notebook-derived CSP/multiband ensemble; retained by later diagnostics | 0.62500 | Final preferred |
| `multiband_balanced` | `{0: 6, 1: 2, 2: 5, 3: 3}` | Earlier balanced multiband hedge; complementary distribution | 0.62500 | Hedge |
| `multiband_best` | `{0: 2, 1: 1, 2: 10, 3: 3}` | Local evidence looked plausible, but distribution was feet-heavy | 0.25000 | Rejected |
| `robust_csp` | `{0: 2, 1: 4, 2: 3, 3: 7}` | Robust diagnostic candidate | 0.50000 | Rejected |
| `model_ceiling_transductive` | `{0: 2, 1: 4, 2: 4, 3: 6}` | RSKF3 F1 0.6250, LOO F1 0.6000 | Not submitted | Rejected; too close to failed robust pattern |
| `model_ceiling_ovo_csp` | `{0: 4, 1: 3, 2: 3, 3: 6}` | RSKF3 F1 0.4340, LOO F1 0.6250 | Not submitted | Rejected; weak repeated CV |
| `model_ceiling_consensus` | `{0: 5, 1: 2, 2: 2, 3: 7}` | RSKF3 F1 0.5556, LOO F1 0.6083 | Not submitted | Rejected; not stronger than current hedge |

## Final Selection Rationale

The final selected Task 1 solution remains the earlier CSP/multiband ensemble family. The later robust and model-ceiling phases improved understanding of the failure modes, but they did not provide enough stable evidence to replace `task1_submission_ensemble_best.csv` or the `task1_submission_multiband_balanced.csv` hedge. In particular, the best model-ceiling candidate resembled the already failed `robust_csp` diagnostic, while the OVO and consensus variants were not stronger than the current 0.625 hedge.

## Risk And Error Analysis

The main risk is the tiny labeled set. With 16 training trials and four classes, validation estimates are unstable and folds are sensitive to individual trials. This made public leaderboard feedback useful only as a guardrail against obvious overfit, not as a target to optimize directly.

The clearest error pattern was feet-heavy overfit. `task1_submission_multiband_best.csv` predicted feet for 10 of 16 test rows and scored only 0.25000 publicly. That failure made extreme class distributions high risk even when local validation looked acceptable.

Public score was used only as a diagnostic guardrail. The workflow did not use public row-flipping, did not infer public labels, and did not tune individual test rows against leaderboard feedback. Later phases explicitly avoided submitting weaker or more speculative candidates once diagnostic evidence showed they were unlikely to beat the existing hedge.

## Reproducibility

- Python: `C:\Users\USER\AppData\Local\Programs\Python\Python314\python.exe`, expected version `Python 3.14.3`.
- Data paths: `part1/task1/data/train.npz` and `part1/task1/data/test.npz`.
- Final notebook path: `part1/task1/task1.ipynb`.
- Diagnostic helpers: `part1/task1/experiments/`.
- Fixed seed used by helper scripts: `SEED = 42`.
- Generated CSV submissions and JSON diagnostics are intentionally uncommitted.
- CSV output format: header exactly `id,label`, 16 rows, preserved test order, labels in `{0, 1, 2, 3}`.
