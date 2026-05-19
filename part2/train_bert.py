"""Fine-tune the Part 2 transformer model.

This script is intentionally separate from part2.ipynb so the stable
classical notebook stays intact. It trains a binary sequence classifier,
reports validation Macro-F1, and writes a Kaggle submission CSV. The default
configuration matches the final selected RoBERTa-base seed 42 workflow.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


TEXT_COL_CANDIDATES = ("text", "content", "utterance", "sentence")
CLASSICAL_TUNED_BASELINE = "0.8805 +/- 0.0262"
FINAL_MODEL_NAME = "roberta-base"
FINAL_MODEL_OOF_MACRO_F1 = 0.968701
FINAL_MODEL_PUBLIC_SCORE = 0.9686
TRANSFORMER_HOLDOUT_BASELINE = 0.9329


def require_transformer_deps():
    """Import heavy dependencies lazily and show a useful install hint."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install the BERT experiment dependencies with:\n"
            "  py -3.14 -m pip install torch transformers tqdm\n"
            "If you have CUDA, install the PyTorch wheel recommended for your CUDA version."
        ) from exc

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "tqdm": tqdm,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-base seed 42 for Part 2 dialogue continuity.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-name", default=FINAL_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "roberta_seed42")
    parser.add_argument("--submission-path", type=Path, default=Path(__file__).resolve().parent / "submission_roberta_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--k-folds", type=int, default=0, help="Optional Stratified K-Fold CV. 0 disables CV.")
    parser.add_argument("--threshold-tune", action="store_true", help="Tune the positive-class probability threshold on validation data.")
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--multi-seed", action="store_true", help="Train multiple holdout models and average test probabilities.")
    parser.add_argument("--multi-seeds", default="13,42,2026", help="Comma-separated seeds used when --multi-seed is enabled.")
    parser.add_argument("--debug", action="store_true", help="Run a tiny end-to-end smoke test.")
    return parser.parse_args()


def set_seed(seed: int, torch_module=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def find_csv(data_dir: Path, name: str) -> Path:
    candidates = [
        data_dir / name,
        Path.cwd() / name,
        Path.cwd() / "part2" / name,
        Path(__file__).resolve().parent / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate {name}. Tried: {[str(p) for p in candidates]}")


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_col = next((col for col in TEXT_COL_CANDIDATES if col in df.columns), None)
    if text_col is None:
        raise KeyError(f"No text column found. Columns: {df.columns.tolist()}")
    df = df.copy()
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = find_csv(data_dir, "train.csv")
    test_path = find_csv(data_dir, "test.csv")
    train_df = standardize_frame(pd.read_csv(train_path))
    test_df = standardize_frame(pd.read_csv(test_path))
    if "label" not in train_df.columns:
        raise KeyError("train.csv must contain a label column.")
    train_df["label"] = train_df["label"].astype(int)
    print(f"train: {train_path.resolve()} shape={train_df.shape}")
    print(f"test : {test_path.resolve()} shape={test_df.shape}")
    print("label distribution:")
    print(train_df["label"].value_counts().sort_index().to_string())
    return train_df, test_df


def make_debug_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n >= len(df):
        return df.reset_index(drop=True)
    _, subset = train_test_split(
        df,
        test_size=n,
        random_state=seed,
        stratify=df["label"],
    )
    return subset.reset_index(drop=True)


def make_datasets_class(dataset_base):
    class TextDataset(dataset_base):
        def __init__(self, texts: Iterable[str], labels: Iterable[int] | None, tokenizer, max_length: int):
            self.texts = list(texts)
            self.labels = None if labels is None else list(labels)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int):
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            if self.labels is not None:
                item["labels"] = self.labels[idx]
            return item

    return TextDataset


def batch_to_device(batch: dict, device, torch_module) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "labels":
            moved[key] = torch_module.as_tensor(value, dtype=torch_module.long, device=device)
        else:
            moved[key] = value.to(device)
    return moved


def predict(model, dataloader, device, torch_module) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    probs = []
    with torch_module.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device, torch_module)
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch_module.argmax(logits, dim=1).detach().cpu().numpy()
            prob = torch_module.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds.append(pred)
            probs.append(prob)
    return np.concatenate(preds), np.concatenate(probs)


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = []
    for part in seed_text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--multi-seeds must contain at least one integer seed.")
    return seeds


def labels_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(int)


def tune_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float]:
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = labels_from_probs(probs, float(threshold))
        macro_f1 = f1_score(y_true, pred, average="macro")
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def write_submission(path: Path, test_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    submission = pd.DataFrame({"id": test_df["id"].values, "label": labels.astype(int)})
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return validate_submission(path, test_df)


def sibling_submission_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def train_one_split(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    tokenizer,
    model_factory,
    text_dataset_cls,
    libs: dict,
    args: argparse.Namespace,
    fold_name: str,
    save_model: bool,
):
    torch = libs["torch"]
    DataLoader = libs["DataLoader"]
    tqdm = libs["tqdm"]
    scheduler_factory = libs["get_linear_schedule_with_warmup"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)

    train_ds = text_dataset_cls(train_df["text"], train_df["label"], tokenizer, args.max_length)
    valid_ds = text_dataset_cls(valid_df["text"], valid_df["label"], tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = scheduler_factory(optimizer, num_warmup_steps=max(1, int(total_steps * 0.06)), num_training_steps=total_steps)

    best_f1 = -1.0
    best_state = None
    best_report = ""
    best_preds = None
    best_probs = None
    best_valid_y = valid_df["label"].to_numpy()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"{fold_name} epoch {epoch}/{args.epochs}", leave=False)
        for batch in loop:
            batch = batch_to_device(batch, device, torch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        valid_pred, valid_probs = predict(model, valid_loader, device, torch)
        valid_y = valid_df["label"].to_numpy()
        macro_f1 = f1_score(valid_y, valid_pred, average="macro")
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"{fold_name} epoch {epoch}: train_loss={avg_loss:.4f} val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_preds = valid_pred
            best_probs = valid_probs
            best_report = classification_report(valid_y, valid_pred, digits=4)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_model:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved best model/tokenizer to {args.output_dir}")

    return model, best_f1, best_report, best_preds, best_probs, best_valid_y


def validate_submission(path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "label"], f"Bad header: {sub.columns.tolist()}"
    assert len(sub) == len(test_df), f"Bad row count: {len(sub)} != {len(test_df)}"
    assert sub["id"].tolist() == test_df["id"].tolist(), "Submission IDs/order do not match test.csv"
    assert set(sub["label"].unique()).issubset({0, 1}), f"Bad labels: {set(sub['label'].unique())}"
    return sub


def run_kfold(train_df, test_df, tokenizer, model_factory, text_dataset_cls, libs, args):
    if args.k_folds < 2:
        return None
    print(f"\nRunning optional {args.k_folds}-fold Stratified CV.")
    torch = libs["torch"]
    DataLoader = libs["DataLoader"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    oof_y = []
    oof_probs = []
    test_probs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], train_df["label"]), start=1):
        fold_args = copy.deepcopy(args)
        fold_args.output_dir = args.output_dir / f"fold_{fold}"
        model, score, _, _, valid_probs, valid_y = train_one_split(
            train_df=train_df.iloc[tr_idx].reset_index(drop=True),
            valid_df=train_df.iloc[va_idx].reset_index(drop=True),
            tokenizer=tokenizer,
            model_factory=model_factory,
            text_dataset_cls=text_dataset_cls,
            libs=libs,
            args=fold_args,
            fold_name=f"fold {fold}",
            save_model=False,
        )
        fold_scores.append(score)
        oof_y.append(valid_y)
        oof_probs.append(valid_probs)

        test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        _, fold_test_probs = predict(model, test_loader, device, torch)
        test_probs.append(fold_test_probs)
        print(f"fold {fold} best val Macro-F1={score:.4f}")
    print(f"K-Fold Macro-F1 mean={np.mean(fold_scores):.4f} std={np.std(fold_scores):.4f} scores={[round(s, 4) for s in fold_scores]}")
    oof_y_arr = np.concatenate(oof_y)
    oof_probs_arr = np.concatenate(oof_probs)
    if args.threshold_tune:
        threshold, threshold_f1 = tune_threshold(
            oof_y_arr,
            oof_probs_arr,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
    else:
        threshold, threshold_f1 = 0.5, f1_score(oof_y_arr, labels_from_probs(oof_probs_arr, 0.5), average="macro")
    mean_test_probs = np.mean(test_probs, axis=0)
    kfold_path = sibling_submission_path(args.submission_path, "kfold")
    checked = write_submission(kfold_path, test_df, labels_from_probs(mean_test_probs, threshold))
    print(f"K-Fold threshold={threshold:.2f} OOF Macro-F1={threshold_f1:.4f}")
    print(f"K-Fold submission saved: {kfold_path}")
    print(f"K-Fold prediction distribution: {checked['label'].value_counts().sort_index().to_dict()}")
    return {
        "scores": fold_scores,
        "threshold": threshold,
        "threshold_f1": threshold_f1,
        "submission_path": kfold_path,
    }


def train_holdout_and_predict(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    model_factory,
    text_dataset_cls,
    libs: dict,
    args: argparse.Namespace,
    seed: int,
    save_model: bool,
    fold_name: str,
):
    torch = libs["torch"]
    DataLoader = libs["DataLoader"]
    set_seed(seed, torch)
    train_split, valid_split = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=seed,
        stratify=train_df["label"],
    )
    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)
    print(f"\n{fold_name}: train={len(train_split)} valid={len(valid_split)} seed={seed}")
    model, best_f1, report, valid_pred, valid_probs, valid_y = train_one_split(
        train_df=train_split,
        valid_df=valid_split,
        tokenizer=tokenizer,
        model_factory=model_factory,
        text_dataset_cls=text_dataset_cls,
        libs=libs,
        args=args,
        fold_name=fold_name,
        save_model=save_model,
    )
    default_threshold = 0.5
    default_f1 = f1_score(valid_y, labels_from_probs(valid_probs, default_threshold), average="macro")
    threshold = default_threshold
    threshold_f1 = default_f1
    if args.threshold_tune:
        threshold, threshold_f1 = tune_threshold(
            valid_y,
            valid_probs,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, test_probs = predict(model, test_loader, device, torch)
    return {
        "model": model,
        "best_f1": float(best_f1),
        "report": report,
        "valid_pred": valid_pred,
        "valid_probs": valid_probs,
        "valid_y": valid_y,
        "default_f1": float(default_f1),
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "test_probs": test_probs,
    }


def main() -> None:
    args = parse_args()
    libs = require_transformer_deps()
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForSequenceClassification = libs["AutoModelForSequenceClassification"]
    DataLoader = libs["DataLoader"]

    set_seed(args.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Training can be slow; use --debug first.")

    train_df, test_df = load_data(args.data_dir)
    if args.debug:
        print("DEBUG mode: using a small stratified train subset and 1 epoch.")
        train_df = make_debug_subset(train_df, n=160, seed=args.seed)
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 8)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    text_dataset_cls = make_datasets_class(libs["Dataset"])

    def model_factory():
        return AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    if args.k_folds >= 2 and not args.debug:
        run_kfold(train_df, test_df, tokenizer, model_factory, text_dataset_cls, libs, args)

    if args.multi_seed and not args.debug:
        seeds = parse_seed_list(args.multi_seeds)
        print(f"\nRunning multi-seed holdout ensemble: seeds={seeds}")
        seed_results = []
        test_probs = []
        thresholds = []
        for seed in seeds:
            seed_args = copy.deepcopy(args)
            seed_args.seed = seed
            seed_args.output_dir = args.output_dir / f"seed_{seed}"
            result = train_holdout_and_predict(
                train_df=train_df,
                test_df=test_df,
                tokenizer=tokenizer,
                model_factory=model_factory,
                text_dataset_cls=text_dataset_cls,
                libs=libs,
                args=seed_args,
                seed=seed,
                save_model=True,
                fold_name=f"seed {seed}",
            )
            seed_results.append(result)
            test_probs.append(result["test_probs"])
            thresholds.append(result["threshold"])
            print(
                f"seed {seed}: val_macro_f1={result['best_f1']:.4f} "
                f"threshold={result['threshold']:.2f} threshold_macro_f1={result['threshold_f1']:.4f}"
            )
        ensemble_probs = np.mean(test_probs, axis=0)
        ensemble_threshold = float(np.mean(thresholds)) if args.threshold_tune else 0.5
        multiseed_path = sibling_submission_path(args.submission_path, "multiseed")
        checked = write_submission(multiseed_path, test_df, labels_from_probs(ensemble_probs, ensemble_threshold))
        print("\n=== Multi-Seed Results ===")
        print(f"Validation Macro-F1 mean={np.mean([r['best_f1'] for r in seed_results]):.4f} std={np.std([r['best_f1'] for r in seed_results]):.4f}")
        print(f"Applied threshold={ensemble_threshold:.2f}")
        print(f"Saved: {multiseed_path}")
        print(f"Prediction distribution: {checked['label'].value_counts().sort_index().to_dict()}")
        print("Format check: OK")
        return

    holdout_result = train_holdout_and_predict(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        model_factory=model_factory,
        text_dataset_cls=text_dataset_cls,
        libs=libs,
        args=args,
        seed=args.seed,
        save_model=True,
        fold_name="holdout",
    )
    best_f1 = holdout_result["best_f1"]
    report = holdout_result["report"]

    print("\n=== Validation Results ===")
    print(f"Validation Macro-F1: {best_f1:.4f}")
    if args.threshold_tune:
        print(f"Threshold-tuned Macro-F1: {holdout_result['threshold_f1']:.4f} at threshold={holdout_result['threshold']:.2f}")
    else:
        print(f"Default threshold Macro-F1: {holdout_result['default_f1']:.4f} at threshold=0.50")
    print(report)
    print(f"Final selected model: RoBERTa-base seed 42, OOF Macro-F1={FINAL_MODEL_OOF_MACRO_F1:.6f}, public={FINAL_MODEL_PUBLIC_SCORE:.4f}")
    print(f"Historical transformer holdout baseline for comparison: {TRANSFORMER_HOLDOUT_BASELINE:.4f}")
    print(f"Classical tuned baseline for comparison: {CLASSICAL_TUNED_BASELINE}")

    threshold = holdout_result["threshold"] if args.threshold_tune else 0.5
    test_pred = labels_from_probs(holdout_result["test_probs"], threshold)
    checked = write_submission(args.submission_path, test_df, test_pred)

    print("\n=== Submission ===")
    print(f"Saved: {args.submission_path}")
    print(f"Applied threshold: {threshold:.2f}")
    print(f"Prediction distribution: {checked['label'].value_counts().sort_index().to_dict()}")
    print("Format check: OK")
    if args.threshold_tune:
        threshold_path = sibling_submission_path(args.submission_path, "threshold")
        checked_threshold = write_submission(threshold_path, test_df, test_pred)
        print(f"Threshold submission saved: {threshold_path}")
        print(f"Threshold prediction distribution: {checked_threshold['label'].value_counts().sort_index().to_dict()}")
    if best_f1 < 0.86:
        print("NOTE: Validation is clearly below the tuned classical baseline; upload only as an exploratory Kaggle probe.")
    elif best_f1 < 0.8805:
        print("NOTE: Validation is below the tuned classical baseline; compare carefully before merging.")
    elif args.threshold_tune and holdout_result["threshold_f1"] > TRANSFORMER_HOLDOUT_BASELINE:
        print("NOTE: Threshold-tuned validation improves over the historical transformer baseline; compare against the final RoBERTa seed 42 record before upload.")
    else:
        print("NOTE: Validation is competitive with the tuned classical baseline; worth a Kaggle public-score check.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
