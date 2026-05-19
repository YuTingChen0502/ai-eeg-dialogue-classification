# HW3 Part 2 Reproducibility Notes

This folder contains the Part 2 notebook and supporting scripts used for the final dialogue continuity workflow. The final selected Part 2 model is tuned RoBERTa-base seed 42, with OOF Macro-F1 0.968701 and Kaggle public score 0.9686.

## Files

- `part2.ipynb`: required Part 2 notebook for the assignment.
- `run_candidate_experiments.py`: classical TF-IDF/SVM candidate and imbalance experiments.
- `train_bert.py`: transformer fine-tuning helper; defaults to the final RoBERTa-base seed 42 setup.
- `run_ultimate_optimization.py`: historical DistilBERT candidate comparison and ensemble diagnostics; not the final selected model.
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
python train_bert.py --data-dir . --model-name roberta-base --seed 42 --threshold-tune --submission-path submission_roberta_seed42.csv
```
