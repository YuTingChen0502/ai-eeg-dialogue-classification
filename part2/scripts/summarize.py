from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from part2_utils import BASELINE_SUBMISSION, PYTHON_EXE, ROOT, load_data, read_meta, validate_submission


def _maybe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _line_model(tag: str) -> str:
    meta_path = ROOT / "outputs" / "oof" / f"{tag}_meta.json"
    if not meta_path.exists():
        return f"- `{tag}`: not run"
    meta = read_meta(meta_path)
    folds = ", ".join(f"{float(x):.4f}" for x in meta.get("fold_macro_f1_default", []))
    return (
        f"- `{tag}`: default `{float(meta['oof_macro_f1_default']):.4f}`, "
        f"tuned `{float(meta['oof_macro_f1_tuned']):.4f}`, threshold `{float(meta['oof_threshold']):.2f}`, "
        f"folds `[{folds}]`, std `{float(meta.get('fold_macro_f1_default_std', 0.0)):.4f}`"
    )


def _read_ablation() -> str:
    path = ROOT / "outputs" / "reports" / "heuristic_splitting_ablation.md"
    if not path.exists():
        return "Heuristic splitting ablation not available."
    return path.read_text(encoding="utf-8")


def _commands() -> str:
    p = f'& "{PYTHON_EXE}"'
    lines = [
        "# Reproduce Final Part 2 Pipeline",
        "",
        "Run from:",
        "",
        "```powershell",
        r"cd C:\Coding\Intro_to_AI\project1-part2\part2",
        "```",
        "",
        "All commands use the dedicated CUDA Python executable:",
        f"`{PYTHON_EXE}`",
        "",
        "```powershell",
        f'{p} -c "import sys; print(sys.version)"',
        f'{p} -c "import torch; print(\'cuda=\', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'CPU\')"',
        f'{p} -c "import pandas, numpy, sklearn, transformers; print(\'core imports OK\')"',
        f'{p} -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'microsoft/deberta-v3-base\'); print(\'DeBERTa tokenizer OK\')"',
        f"{p} scripts\\preflight.py",
        f"{p} scripts\\heuristic_split.py",
        f"{p} scripts\\run_oof.py --model-name distilbert-base-uncased --model-tag distilbert_no_aug --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune",
        f"{p} scripts\\run_oof.py --model-name distilbert-base-uncased --model-tag distilbert --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune --augment-heuristic-splitting",
        f"{p} scripts\\run_oof.py --model-name microsoft/deberta-v3-base --model-tag deberta_v3 --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune",
        f"{p} scripts\\run_oof.py --model-name roberta-base --model-tag roberta --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune",
        f"{p} scripts\\classical_oof.py",
        f"{p} scripts\\endpoint_features.py",
        f"{p} scripts\\retrieval_features.py",
        f"{p} scripts\\stack_oof.py",
        f"{p} scripts\\duplicate_transfer.py",
        f"{p} scripts\\summarize.py",
        f"{p} scripts\\update_notebook_summary.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    train_df, test_df = load_data()
    final_path = ROOT / "submissions" / "submission_final.csv"
    stacker_path = ROOT / "submissions" / "submission_stacker_oof.csv"
    baseline_path = ROOT / BASELINE_SUBMISSION
    final_validation = validate_submission(final_path, test_df) if final_path.exists() else None
    selected_path = final_path if final_path.exists() else stacker_path
    selected_validation = validate_submission(selected_path, test_df) if selected_path.exists() else None

    stacker_meta = _maybe_json(ROOT / "outputs" / "stacker_metadata.json") or {}
    duplicate_meta = _maybe_json(ROOT / "outputs" / "duplicate_transfer_metadata.json") or {}
    classical_meta = _maybe_json(ROOT / "outputs" / "features" / "classical_meta.json") or {}
    retrieval_meta = _maybe_json(ROOT / "outputs" / "features" / "retrieval_meta.json") or {}
    endpoint_corr = ROOT / "outputs" / "features" / "endpoint_correlations.csv"

    hamming = ""
    if selected_path.exists() and baseline_path.exists():
        selected = pd.read_csv(selected_path)
        baseline = pd.read_csv(baseline_path)
        hamming_val = int(np.sum(selected["label"].values != baseline["label"].values))
        hamming = f"- Hamming distance vs rollback baseline: `{hamming_val}`"
    recommendation = "Submit `submissions/submission_final.csv` as the final candidate while keeping the rollback baseline available."
    if stacker_meta.get("fallback_used"):
        recommendation = (
            "Submit `submissions/submission_final.csv`; stacker fallback selected the best single OOF model, "
            "so the final candidate stays conservative."
        )

    lines = [
        "# Final Phase Summary",
        "",
        "## Baseline",
        f"- File: `{BASELINE_SUBMISSION}`",
        "- Common-split Macro-F1: `0.9453`",
        "- Threshold: `0.48`",
        "- Distribution: `{0: 217, 1: 283}`",
        "",
        "## Preflight",
        f"- Python executable: `{PYTHON_EXE}`",
        f"- train.csv shape: `{tuple(train_df.shape)}`",
        f"- test.csv shape: `{tuple(test_df.shape)}`",
        f"- train label distribution: `{train_df['label'].value_counts().sort_index().astype(int).to_dict()}`",
        "",
        "## Heuristic Splitting Ablation",
        _read_ablation(),
        "",
        "## Transformer OOF Results",
        _line_model("distilbert_no_aug"),
        _line_model("distilbert"),
        _line_model("deberta_v3"),
        _line_model("roberta"),
        "",
        "## Classical TF-IDF/SVM",
        f"- Meta: `{classical_meta}`",
        "",
        "## Endpoint Features",
        f"- Correlations file: `{endpoint_corr}`",
        "",
        "## Retrieval Features",
        f"- Meta: `{retrieval_meta}`",
        "",
        "## Stacker",
        f"- Metadata: `{stacker_meta}`",
        "",
        "## Duplicate Transfer",
        f"- Metadata: `{duplicate_meta}`",
        "",
        "## Final Selection",
        f"- Final selected file: `{selected_path.relative_to(ROOT) if selected_path.exists() else 'missing'}`",
        f"- CSV validation: `{selected_validation}`",
        hamming,
        f"- Recommendation: {recommendation}",
        "",
        "No test labels, LLM labels, pseudo-labeling, or public leaderboard tuning were used.",
    ]
    (ROOT / "outputs" / "phase_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "outputs" / "REPRODUCE.md").write_text(_commands(), encoding="utf-8")
    snippet = (
        "The final Part 2 system extends the required stable TF-IDF + SVM baseline with an auditable heuristic "
        "splitting ablation as the advanced balancing experiment; because it hurt OOF Macro-F1, final transformer "
        "OOF probabilities use the no-augmentation folds from fine-tuned DistilBERT, DeBERTa-v3-base, and RoBERTa-base. "
        "Classical TF-IDF/SVM scores, endpoint linguistic cues, and leakage-safe retrieval features "
        "are combined in an OOF LogisticRegression stacker with threshold tuning performed only on OOF predictions. "
        "A conservative exact-duplicate consistency check is applied only when a test text exactly matches a unanimous "
        "training group with at least two examples. No test labels, LLM labeling, pseudo-labeling, external labeled "
        "datasets, public leaderboard probing, or leaderboard-based tuning are used."
    )
    (ROOT / "outputs" / "final_report_snippet.md").write_text(snippet + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'outputs' / 'phase_summary.md'}")
    print(f"wrote {ROOT / 'outputs' / 'REPRODUCE.md'}")
    print(f"wrote {ROOT / 'outputs' / 'final_report_snippet.md'}")


if __name__ == "__main__":
    main()
