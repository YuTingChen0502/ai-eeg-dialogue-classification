# Phase 0 Preflight

## Environment
- Python executable: `C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe`
- Python version: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- Torch version: `2.11.0+cu128`
- CUDA available: `True`
- CUDA device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Core imports: pandas, numpy, sklearn, transformers OK
- DeBERTa tokenizer: OK (`microsoft/deberta-v3-base`)
- Tokenizer class: `DebertaV2Tokenizer`

## Data
- train.csv shape: `(1302, 3)`
- test.csv shape: `(500, 2)`
- train columns after standardization: `['id', 'text', 'label']`
- test columns after standardization: `['id', 'text']`
- train label distribution: `{0: 465, 1: 837}`

## Rollback Baseline
- Baseline file: `submission_bert_common_top3_seed_average.csv`
- Backup file: `submissions/backup/submission_bert_common_top3_seed_average.csv`
- Baseline validation: `{'path': 'C:\\Coding\\Intro_to_AI\\project1-part2\\part2\\submission_bert_common_top3_seed_average.csv', 'valid': True, 'errors': [], 'shape': [500, 2], 'distribution': {0: 217, 1: 283}}`

## Policy Checks
- train.csv and test.csv were not modified.
- No test labels, pseudo-labeling, LLM labels, or public leaderboard signals were used.
