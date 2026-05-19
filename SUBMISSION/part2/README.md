# HW3 Part 2 Reproducibility Notes

This folder contains the Part 2 notebook and supporting scripts used for the final dialogue continuity workflow.

## Files

- `part2.ipynb`: required Part 2 notebook for the assignment.
- `run_candidate_experiments.py`: classical TF-IDF/SVM candidate and imbalance experiments.
- `train_bert.py`: transformer fine-tuning helper for DistilBERT/RoBERTa-style experiments.
- `run_ultimate_optimization.py`: late-stage candidate comparison and ensemble diagnostics.
- `requirements.txt`: Python package list for running the notebook and scripts.

## Data

The assignment data and generated CSV submissions are intentionally not included in Git. To reproduce locally, place the official Part 2 `train.csv` and `test.csv` in this folder, then run the notebook or scripts from this directory.

## Basic Commands

```powershell
pip install -r requirements.txt
jupyter notebook part2.ipynb
```

Optional script examples:

```powershell
python run_candidate_experiments.py --data-dir . --output-dir .
python train_bert.py --data-dir .
```
