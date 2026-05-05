"""Fine-tune a lightweight transformer for HW3 Part 2.

This script is intentionally separate from part2.ipynb so the stable
classical notebook stays intact. It trains a binary sequence classifier,
reports validation Macro-F1, and writes a Kaggle submission CSV.
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
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Part 2 dialogue continuity.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "bert")
    parser.add_argument("--submission-path", type=Path, default=Path(__file__).resolve().parent / "submission_bert.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--k-folds", type=int, default=0, help="Optional Stratified K-Fold CV. 0 disables CV.")
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

        valid_pred, _ = predict(model, valid_loader, device, torch)
        valid_y = valid_df["label"].to_numpy()
        macro_f1 = f1_score(valid_y, valid_pred, average="macro")
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"{fold_name} epoch {epoch}: train_loss={avg_loss:.4f} val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_preds = valid_pred
            best_report = classification_report(valid_y, valid_pred, digits=4)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_model:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved best model/tokenizer to {args.output_dir}")

    return model, best_f1, best_report, best_preds


def validate_submission(path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "label"], f"Bad header: {sub.columns.tolist()}"
    assert len(sub) == len(test_df), f"Bad row count: {len(sub)} != {len(test_df)}"
    assert sub["id"].tolist() == test_df["id"].tolist(), "Submission IDs/order do not match test.csv"
    assert set(sub["label"].unique()).issubset({0, 1}), f"Bad labels: {set(sub['label'].unique())}"
    return sub


def run_kfold(train_df, tokenizer, model_factory, text_dataset_cls, libs, args):
    if args.k_folds < 2:
        return None
    print(f"\nRunning optional {args.k_folds}-fold Stratified CV.")
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], train_df["label"]), start=1):
        _, score, _, _ = train_one_split(
            train_df=train_df.iloc[tr_idx].reset_index(drop=True),
            valid_df=train_df.iloc[va_idx].reset_index(drop=True),
            tokenizer=tokenizer,
            model_factory=model_factory,
            text_dataset_cls=text_dataset_cls,
            libs=libs,
            args=args,
            fold_name=f"fold {fold}",
            save_model=False,
        )
        fold_scores.append(score)
        print(f"fold {fold} best val Macro-F1={score:.4f}")
    print(f"K-Fold Macro-F1 mean={np.mean(fold_scores):.4f} std={np.std(fold_scores):.4f} scores={[round(s, 4) for s in fold_scores]}")
    return fold_scores


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
        run_kfold(train_df, tokenizer, model_factory, text_dataset_cls, libs, args)

    train_split, valid_split = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train_df["label"],
    )
    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)

    print(f"\nHoldout split: train={len(train_split)} valid={len(valid_split)}")
    model, best_f1, report, _ = train_one_split(
        train_df=train_split,
        valid_df=valid_split,
        tokenizer=tokenizer,
        model_factory=model_factory,
        text_dataset_cls=text_dataset_cls,
        libs=libs,
        args=args,
        fold_name="holdout",
        save_model=True,
    )

    print("\n=== Validation Results ===")
    print(f"Validation Macro-F1: {best_f1:.4f}")
    print(report)
    print(f"Classical tuned baseline for comparison: {CLASSICAL_TUNED_BASELINE}")

    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_pred, _ = predict(model, test_loader, device, torch)

    submission = pd.DataFrame({"id": test_df["id"].values, "label": test_pred.astype(int)})
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    checked = validate_submission(args.submission_path, test_df)

    print("\n=== Submission ===")
    print(f"Saved: {args.submission_path}")
    print(f"Prediction distribution: {checked['label'].value_counts().sort_index().to_dict()}")
    print("Format check: OK")
    if best_f1 < 0.86:
        print("NOTE: Validation is clearly below the tuned classical baseline; upload only as an exploratory Kaggle probe.")
    elif best_f1 < 0.8805:
        print("NOTE: Validation is below the tuned classical baseline; compare carefully before merging.")
    else:
        print("NOTE: Validation is competitive with the tuned classical baseline; worth a Kaggle public-score check.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
