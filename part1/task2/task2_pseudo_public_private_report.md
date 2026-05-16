# Task 2 pseudo-public/private validation

## Final Task 2 Status

`chen-experiment/task2` is the preserved final Task 2 branch. The task is cross-subject EEG classification, so leave-one-subject-out validation and subject-shift diagnostics are treated as the main local evidence.

The strongest historical public anchors remain:

- `submission_task2_prior_public_adjusted.csv`: distribution `{0:6,1:8,2:9,3:9}`, public `0.68750`.
- `submission_task2_prior_class2_less.csv`: distribution `{0:6,1:9,2:8,3:9}`, public `0.68750`.

The final diagnostic work emphasizes LOSO validation, tangent/covariance/CSP-related variants, and pseudo-public/private simulation on held-out training subjects. Class 2 remains the weakest class in the diagnostics. Public feedback was used only as a sparse guardrail; no external data, pretrained models, inferred public labels, or real test labels were used for training.

Oracle pseudo-public methods below use labels only inside held-out training-subject simulation.
They are not legal real-test procedures.

- held-out subjects: 10
- random splits per subject: 6
- pseudo-public size: 16
- configs: tangent_broad_lda, cov_align_subject_lda_class2_up
- model bundles: tangent_broad_lda, cov_align_subject_lda_class2_up, ensemble_tangent_cov2
- execution note: an initial 24-split run timed out before producing final artifacts; the verified run below uses 6 splits per subject.

| model | method | legal | macro F1 | acc | class F1 0/1/2/3 | worst subj | win | mean delta | median delta | cats | mean changed | pred dist |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| cov_align_subject_lda_class2_up | baseline_private_only | yes | 0.5457+/-0.1406 | 0.5594+/-0.1400 | 0.5130/0.5523/0.5076/0.6097 | 0.4121 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | {"0": 275, "1": 239, "2": 217, "3": 229} |
| cov_align_subject_lda_class2_up | unlabeled_batch32_norm | yes | 0.5617+/-0.1327 | 0.5781+/-0.1279 | 0.5245/0.5726/0.5223/0.6272 | 0.4270 | 0.5167 | 0.0160 | 0.0013 | 14 | 2.3500 | {"0": 257, "1": 250, "2": 225, "3": 228} |
| cov_align_subject_lda_class2_up | unlabeled_em_prior32 | yes | 0.5617+/-0.1327 | 0.5781+/-0.1279 | 0.5245/0.5726/0.5223/0.6272 | 0.4270 | 0.5167 | 0.0160 | 0.0013 | 14 | 2.3500 | {"0": 257, "1": 250, "2": 225, "3": 228} |
| cov_align_subject_lda_class2_up | unlabeled_uniform_quota32 | yes | 0.5605+/-0.1432 | 0.5740+/-0.1437 | 0.4786/0.5575/0.5268/0.6790 | 0.4393 | 0.5167 | 0.0148 | 0.0030 | 17 | 2.8833 | {"0": 225, "1": 232, "2": 252, "3": 251} |
| cov_align_subject_lda_class2_up | unlabeled_conservative_quota32 | yes | 0.5577+/-0.1344 | 0.5750+/-0.1295 | 0.5245/0.5726/0.5116/0.6222 | 0.4270 | 0.5000 | 0.0121 | 0.0002 | 16 | 2.3667 | {"0": 257, "1": 250, "2": 222, "3": 231} |
| cov_align_subject_lda_class2_up | oracle_public_bias_grid | oracle | 0.5596+/-0.1328 | 0.5750+/-0.1290 | 0.5225/0.5704/0.5205/0.6249 | 0.4248 | 0.5167 | 0.0139 | 0.0039 | 13 | 2.3500 | {"0": 261, "1": 253, "2": 224, "3": 222} |
| cov_align_subject_lda_class2_up | oracle_public_prior_ratio | oracle | 0.5555+/-0.1322 | 0.5740+/-0.1278 | 0.5229/0.5648/0.5167/0.6178 | 0.4270 | 0.4667 | 0.0099 | 0.0000 | 18 | 2.4500 | {"0": 258, "1": 249, "2": 224, "3": 229} |
| cov_align_subject_lda_class2_up | oracle_public_confusion | oracle | 0.3876+/-0.1642 | 0.4344+/-0.1571 | 0.3793/0.3713/0.3559/0.4441 | 0.1993 | 0.1333 | -0.1580 | -0.1409 | 44 | 6.3500 | {"0": 289, "1": 232, "2": 243, "3": 196} |
| ensemble_tangent_cov2 | baseline_private_only | yes | 0.5445+/-0.1382 | 0.5604+/-0.1350 | 0.5246/0.5608/0.4800/0.6125 | 0.4414 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | {"0": 264, "1": 251, "2": 215, "3": 230} |
| ensemble_tangent_cov2 | unlabeled_conservative_quota32 | yes | 0.5565+/-0.1534 | 0.5708+/-0.1490 | 0.5727/0.5586/0.4845/0.6102 | 0.4221 | 0.5000 | 0.0120 | 0.0004 | 18 | 2.8667 | {"0": 231, "1": 227, "2": 253, "3": 249} |
| ensemble_tangent_cov2 | unlabeled_uniform_quota32 | yes | 0.5521+/-0.1506 | 0.5646+/-0.1452 | 0.5525/0.5521/0.4888/0.6151 | 0.4221 | 0.4833 | 0.0077 | 0.0000 | 16 | 3.1500 | {"0": 221, "1": 233, "2": 260, "3": 246} |
| ensemble_tangent_cov2 | unlabeled_batch32_norm | yes | 0.5418+/-0.1512 | 0.5646+/-0.1434 | 0.5541/0.5611/0.4495/0.6025 | 0.4004 | 0.3833 | -0.0027 | 0.0000 | 16 | 1.9167 | {"0": 261, "1": 242, "2": 219, "3": 238} |
| ensemble_tangent_cov2 | unlabeled_em_prior32 | yes | 0.5374+/-0.1573 | 0.5760+/-0.1466 | 0.5623/0.5132/0.3944/0.6796 | 0.3772 | 0.4833 | -0.0071 | -0.0074 | 20 | 3.2000 | {"0": 268, "1": 241, "2": 183, "3": 268} |
| ensemble_tangent_cov2 | oracle_public_bias_grid | oracle | 0.5290+/-0.1591 | 0.5563+/-0.1478 | 0.5577/0.5247/0.4233/0.6105 | 0.4112 | 0.4333 | -0.0154 | -0.0044 | 23 | 3.6167 | {"0": 288, "1": 219, "2": 208, "3": 245} |
| ensemble_tangent_cov2 | oracle_public_prior_ratio | oracle | 0.5256+/-0.1517 | 0.5469+/-0.1517 | 0.5477/0.5186/0.4455/0.5907 | 0.3817 | 0.4000 | -0.0188 | -0.0171 | 23 | 3.7333 | {"0": 248, "1": 212, "2": 257, "3": 243} |
| ensemble_tangent_cov2 | oracle_public_confusion | oracle | 0.3745+/-0.1480 | 0.4302+/-0.1475 | 0.3339/0.3822/0.2925/0.4895 | 0.2891 | 0.1500 | -0.1699 | -0.1515 | 43 | 6.7167 | {"0": 279, "1": 205, "2": 249, "3": 227} |
| tangent_broad_lda | baseline_private_only | yes | 0.5263+/-0.1612 | 0.5396+/-0.1552 | 0.5844/0.5220/0.4124/0.5864 | 0.3227 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | {"0": 245, "1": 238, "2": 224, "3": 253} |
| tangent_broad_lda | unlabeled_conservative_quota32 | yes | 0.5482+/-0.1627 | 0.5615+/-0.1579 | 0.6103/0.5604/0.4136/0.6085 | 0.3455 | 0.5167 | 0.0219 | 0.0049 | 8 | 2.0000 | {"0": 233, "1": 231, "2": 230, "3": 266} |
| tangent_broad_lda | unlabeled_em_prior32 | yes | 0.5476+/-0.1618 | 0.5667+/-0.1593 | 0.5981/0.5561/0.4141/0.6220 | 0.3705 | 0.4500 | 0.0213 | 0.0000 | 9 | 2.1000 | {"0": 229, "1": 234, "2": 222, "3": 275} |
| tangent_broad_lda | unlabeled_batch32_norm | yes | 0.5473+/-0.1672 | 0.5667+/-0.1645 | 0.5864/0.5558/0.4221/0.6249 | 0.3705 | 0.5000 | 0.0210 | 0.0067 | 9 | 2.1667 | {"0": 237, "1": 230, "2": 223, "3": 270} |
| tangent_broad_lda | unlabeled_uniform_quota32 | yes | 0.5242+/-0.1389 | 0.5344+/-0.1357 | 0.5772/0.5457/0.4220/0.5521 | 0.3468 | 0.4500 | -0.0021 | 0.0000 | 20 | 2.6833 | {"0": 221, "1": 236, "2": 259, "3": 244} |
| tangent_broad_lda | oracle_public_prior_ratio | oracle | 0.5348+/-0.1618 | 0.5521+/-0.1582 | 0.5719/0.5483/0.4064/0.6127 | 0.3568 | 0.4667 | 0.0085 | 0.0000 | 15 | 2.1667 | {"0": 232, "1": 235, "2": 229, "3": 264} |
| tangent_broad_lda | oracle_public_bias_grid | oracle | 0.5273+/-0.1619 | 0.5500+/-0.1580 | 0.5494/0.5311/0.4108/0.6180 | 0.3836 | 0.4167 | 0.0010 | 0.0000 | 15 | 2.3500 | {"0": 234, "1": 236, "2": 227, "3": 263} |
| tangent_broad_lda | oracle_public_confusion | oracle | 0.4073+/-0.1731 | 0.4594+/-0.1653 | 0.4419/0.3649/0.3170/0.5052 | 0.2277 | 0.2500 | -0.1190 | -0.1006 | 40 | 6.2167 | {"0": 259, "1": 188, "2": 262, "3": 251} |

## Candidate diagnostics

No real-test candidate was emitted because no legal method cleared the configured validation threshold.
