# Heuristic Splitting Ablation

| tag | augmentation | tuned OOF Macro-F1 | threshold | distribution |
| --- | --- | ---: | ---: | --- |
| distilbert_no_aug | no | 0.9531 | 0.62 | {'0': 463, '1': 839} |
| distilbert | yes | 0.9121 | 0.30 | {'0': 529, '1': 773} |

Delta (aug - no_aug): `-0.0410`.
Decision: **heuristic splitting clearly hurt DistilBERT OOF; disable for later final transformer runs**.

Thresholds were tuned only on OOF probabilities. No test labels or public leaderboard signals were used.
