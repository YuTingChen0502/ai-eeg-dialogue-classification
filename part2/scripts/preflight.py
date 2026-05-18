from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from part2_utils import BASELINE_SUBMISSION, ROOT, ensure_dirs, format_dist, load_data, validate_submission


def main() -> None:
    ensure_dirs()
    train_df, test_df = load_data()
    baseline_path = ROOT / BASELINE_SUBMISSION
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing rollback baseline: {baseline_path}")
    backup_path = ROOT / "submissions" / "backup" / BASELINE_SUBMISSION
    if not backup_path.exists():
        shutil.copy2(baseline_path, backup_path)
    baseline_validation = validate_submission(baseline_path, test_df)
    validate_submission(backup_path, test_df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; stopping per project constraints.")

    report = [
        "# Phase 0 Preflight",
        "",
        "## Environment",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{sys.version}`",
        f"- Torch version: `{torch.__version__}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        f"- CUDA device: `{device_name}`",
        "- Core imports: pandas, numpy, sklearn, transformers OK",
        "- DeBERTa tokenizer: OK (`microsoft/deberta-v3-base`)",
        f"- Tokenizer class: `{tokenizer.__class__.__name__}`",
        "",
        "## Data",
        f"- train.csv shape: `{tuple(train_df.shape)}`",
        f"- test.csv shape: `{tuple(test_df.shape)}`",
        f"- train columns after standardization: `{list(train_df.columns)}`",
        f"- test columns after standardization: `{list(test_df.columns)}`",
        f"- train label distribution: `{format_dist(train_df['label'].value_counts())}`",
        "",
        "## Rollback Baseline",
        f"- Baseline file: `{BASELINE_SUBMISSION}`",
        f"- Backup file: `submissions/backup/{BASELINE_SUBMISSION}`",
        f"- Baseline validation: `{baseline_validation}`",
        "",
        "## Policy Checks",
        "- train.csv and test.csv were not modified.",
        "- No test labels, pseudo-labeling, LLM labels, or public leaderboard signals were used.",
    ]
    out = ROOT / "outputs" / "reports" / "preflight.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"python={sys.version}")
    print(f"cuda={torch.cuda.is_available()} device={device_name}")
    print(f"train shape={train_df.shape} test shape={test_df.shape}")
    print(f"label distribution={train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"rollback validation valid={baseline_validation['valid']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
