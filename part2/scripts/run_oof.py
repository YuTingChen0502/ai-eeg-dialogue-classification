from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from heuristic_split import augment_with_splits
from part2_utils import ROOT, assert_finite, format_dist, labels_from_probs, load_data, tune_threshold, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune transformer OOF probabilities for Part 2.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threshold-tune", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--augment-heuristic-splitting", action="store_true")
    parser.add_argument("--n-aug-per-complete", type=int, default=1)
    parser.add_argument("--min-keep-tokens", type=int, default=3)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def require_libs():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "tqdm": tqdm,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = False


def make_dataset_cls(dataset_base):
    class TextDataset(dataset_base):
        def __init__(self, texts, labels, tokenizer, max_length: int):
            self.texts = list(texts)
            self.labels = None if labels is None else list(labels)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            if self.labels is not None:
                item["labels"] = int(self.labels[idx])
            return item

    return TextDataset


def batch_to_device(batch: dict, device, torch_module) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "labels":
            moved[key] = torch_module.as_tensor(value, dtype=torch_module.long, device=device)
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def predict_probs(model, loader, device, torch_module) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch_module.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device, torch_module)
            outputs = model(**batch)
            prob = torch_module.softmax(outputs.logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs)


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def fold_attempts(requested_batch: int, requested_max_length: int) -> list[tuple[int, int, int, str]]:
    attempts: list[tuple[int, int, int, str]] = []
    for batch_size in [requested_batch, 8, 4]:
        if batch_size <= requested_batch and all(batch_size != item[0] for item in attempts):
            grad_accum = max(1, requested_batch // batch_size)
            attempts.append((batch_size, requested_max_length, grad_accum, "requested" if batch_size == requested_batch else f"batch_{batch_size}"))
    if requested_max_length > 96:
        attempts.append((4, 96, max(1, requested_batch // 4), "batch_4_maxlen_96"))
    return attempts


def train_fold(
    *,
    args: argparse.Namespace,
    libs: dict,
    tokenizer,
    text_dataset_cls,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_index: int,
    batch_size: int,
    max_length: int,
    grad_accum_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch = libs["torch"]
    DataLoader = libs["DataLoader"]
    tqdm = libs["tqdm"]
    AutoModel = libs["AutoModelForSequenceClassification"]
    scheduler_factory = libs["get_linear_schedule_with_warmup"]

    set_seed(args.seed + fold_index, torch)
    device = torch.device("cuda")
    # Some local HF caches may contain half-precision weights. Train in fp32
    # parameters and let autocast handle fp16 activations when enabled.
    model = AutoModel.from_pretrained(args.model_name, num_labels=2).float().to(device)

    train_ds = text_dataset_cls(train_df["text"].tolist(), train_df["label"].astype(int).tolist(), tokenizer, max_length)
    valid_ds = text_dataset_cls(valid_df["text"].tolist(), valid_df["label"].astype(int).tolist(), tokenizer, max_length)
    test_ds = text_dataset_cls(test_df["text"].tolist(), None, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = int(np.ceil(len(train_loader) / grad_accum_steps))
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = scheduler_factory(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    use_amp = torch.cuda.is_available() and not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epoch_losses: list[float] = []
    epoch_valid_f1: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        step_count = 0
        progress = tqdm(train_loader, desc=f"{args.model_tag} fold {fold_index + 1} epoch {epoch + 1}", leave=False)
        for step, batch in enumerate(progress):
            batch = batch_to_device(batch, device, torch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * grad_accum_steps
            is_update = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
        valid_probs_epoch = predict_probs(model, valid_loader, device, torch)
        valid_pred_epoch = labels_from_probs(valid_probs_epoch, 0.5)
        valid_f1_epoch = f1_score(valid_df["label"].values, valid_pred_epoch, average="macro")
        epoch_losses.append(running / max(1, len(train_loader)))
        epoch_valid_f1.append(float(valid_f1_epoch))
        print(
            f"fold {fold_index + 1} epoch {epoch + 1}: "
            f"loss={epoch_losses[-1]:.4f} valid_macro_f1@0.5={valid_f1_epoch:.4f}"
        )

    valid_probs = predict_probs(model, valid_loader, device, torch)
    test_probs = predict_probs(model, test_loader, device, torch)
    diagnostics = {
        "batch_size": batch_size,
        "max_length": max_length,
        "grad_accum_steps": grad_accum_steps,
        "epoch_losses": epoch_losses,
        "epoch_valid_macro_f1_default": epoch_valid_f1,
    }

    del model, train_loader, valid_loader, test_loader, train_ds, valid_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()
    return valid_probs, test_probs, diagnostics


def maybe_write_ablation_report(output_dir: Path) -> None:
    no_aug_path = output_dir / "distilbert_no_aug_meta.json"
    aug_path = output_dir / "distilbert_meta.json"
    if not no_aug_path.exists() or not aug_path.exists():
        return
    import json

    no_aug = json.loads(no_aug_path.read_text(encoding="utf-8"))
    aug = json.loads(aug_path.read_text(encoding="utf-8"))
    no_aug_f1 = float(no_aug["oof_macro_f1_tuned"])
    aug_f1 = float(aug["oof_macro_f1_tuned"])
    delta = aug_f1 - no_aug_f1
    decision = "keep heuristic splitting for final models"
    if delta < -0.005:
        decision = "heuristic splitting clearly hurt DistilBERT OOF; disable for later final transformer runs"
    report = [
        "# Heuristic Splitting Ablation",
        "",
        "| tag | augmentation | tuned OOF Macro-F1 | threshold | distribution |",
        "| --- | --- | ---: | ---: | --- |",
        f"| distilbert_no_aug | no | {no_aug_f1:.4f} | {float(no_aug['oof_threshold']):.2f} | {no_aug['prediction_distribution_at_threshold']} |",
        f"| distilbert | yes | {aug_f1:.4f} | {float(aug['oof_threshold']):.2f} | {aug['prediction_distribution_at_threshold']} |",
        "",
        f"Delta (aug - no_aug): `{delta:.4f}`.",
        f"Decision: **{decision}**.",
        "",
        "Thresholds were tuned only on OOF probabilities. No test labels or public leaderboard signals were used.",
    ]
    (ROOT / "outputs" / "reports" / "heuristic_splitting_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    libs = require_libs()
    torch = libs["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; stopping per constraints.")
    set_seed(args.seed, torch)
    train_df, test_df = load_data()
    y = train_df["label"].values.astype(int)
    output_dir = ROOT / "outputs" / "oof"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = libs["AutoTokenizer"].from_pretrained(args.model_name)
    text_dataset_cls = make_dataset_cls(libs["Dataset"])
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(train_df), dtype=np.float32)
    test_fold_probs: list[np.ndarray] = []
    fold_rows: list[dict] = []
    fold_scores: list[float] = []
    fold_diagnostics: list[dict] = []
    oom_fallbacks: list[dict] = []

    print(f"model={args.model_name} tag={args.model_tag}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"train shape={train_df.shape} test shape={test_df.shape}")
    print(f"label distribution={train_df['label'].value_counts().sort_index().to_dict()}")

    for fold_index, (train_idx, valid_idx) in enumerate(skf.split(train_df["text"], y)):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        original_train_size = len(fold_train)
        if args.augment_heuristic_splitting:
            fold_train = augment_with_splits(
                fold_train,
                rng_seed=args.seed + fold_index,
                n_aug_per_complete=args.n_aug_per_complete,
                min_keep_tokens=args.min_keep_tokens,
                text_col="text",
            )
        print(
            f"fold {fold_index + 1}/{args.k_folds}: "
            f"train_original={original_train_size} train_used={len(fold_train)} valid={len(fold_valid)} "
            f"train_dist={fold_train['label'].value_counts().sort_index().to_dict()}"
        )

        last_exc: BaseException | None = None
        for batch_size, max_length, grad_accum, note in fold_attempts(args.batch_size, args.max_length):
            try:
                valid_probs, test_probs, diagnostics = train_fold(
                    args=args,
                    libs=libs,
                    tokenizer=tokenizer,
                    text_dataset_cls=text_dataset_cls,
                    train_df=fold_train,
                    valid_df=fold_valid,
                    test_df=test_df,
                    fold_index=fold_index,
                    batch_size=batch_size,
                    max_length=max_length,
                    grad_accum_steps=grad_accum,
                )
                diagnostics["oom_attempt_note"] = note
                if note != "requested":
                    oom_fallbacks.append({"fold": fold_index + 1, "fallback": note})
                break
            except RuntimeError as exc:
                last_exc = exc
                if not is_oom(exc):
                    raise
                print(f"OOM on fold {fold_index + 1} with batch={batch_size}, max_length={max_length}; trying fallback.")
                gc.collect()
                torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"All OOM fallbacks failed for fold {fold_index + 1}") from last_exc

        valid_probs = np.asarray(valid_probs, dtype=np.float32)
        test_probs = np.asarray(test_probs, dtype=np.float32)
        assert_finite(f"{args.model_tag} fold valid probs", valid_probs)
        assert_finite(f"{args.model_tag} fold test probs", test_probs)
        oof_probs[valid_idx] = valid_probs
        test_fold_probs.append(test_probs)
        fold_f1 = f1_score(y[valid_idx], labels_from_probs(valid_probs, 0.5), average="macro")
        fold_scores.append(float(fold_f1))
        diagnostics.update(
            {
                "fold": fold_index + 1,
                "valid_macro_f1_default": float(fold_f1),
                "original_train_size": int(original_train_size),
                "used_train_size": int(len(fold_train)),
                "valid_size": int(len(fold_valid)),
            }
        )
        fold_diagnostics.append(diagnostics)
        for local_pos, orig_idx in enumerate(valid_idx):
            fold_rows.append(
                {
                    "row_index": int(orig_idx),
                    "id": train_df.iloc[orig_idx]["id"],
                    "fold": fold_index + 1,
                    "label": int(y[orig_idx]),
                    "prob": float(valid_probs[local_pos]),
                    "pred_default": int(valid_probs[local_pos] >= 0.5),
                }
            )
        print(f"fold {fold_index + 1} Macro-F1@0.5={fold_f1:.4f}")
        if args.model_tag.startswith("deberta") and fold_index == 0 and fold_f1 < 0.88:
            raise SystemExit("First DeBERTa fold Macro-F1 < 0.88; stopping to inspect setup.")

    test_probs_mean = np.mean(np.vstack(test_fold_probs), axis=0).astype(np.float32)
    assert len(oof_probs) == len(train_df)
    assert len(test_probs_mean) == len(test_df)
    assert_finite(f"{args.model_tag} OOF probs", oof_probs)
    assert_finite(f"{args.model_tag} test probs", test_probs_mean)

    oof_default = f1_score(y, labels_from_probs(oof_probs, 0.5), average="macro")
    if args.threshold_tune:
        threshold, tuned_f1 = tune_threshold(
            y,
            oof_probs,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
    else:
        threshold, tuned_f1 = 0.5, oof_default
    pred_tuned = labels_from_probs(oof_probs, threshold)
    pred_dist = pd.Series(pred_tuned).value_counts().sort_index().astype(int).to_dict()
    print(f"OOF Macro-F1@0.5={oof_default:.4f}")
    print(f"OOF tuned threshold={threshold:.2f} Macro-F1={tuned_f1:.4f}")
    print(f"OOF tuned distribution={format_dist(pred_dist)}")

    np.save(output_dir / f"{args.model_tag}_oof.npy", oof_probs)
    np.save(output_dir / f"{args.model_tag}_test.npy", test_probs_mean)
    folds_df = pd.DataFrame(fold_rows).sort_values("row_index").reset_index(drop=True)
    folds_df.to_csv(output_dir / f"{args.model_tag}_folds.csv", index=False)
    meta = {
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "seed": args.seed,
        "k_folds": args.k_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fp16_enabled": not args.no_fp16,
        "augmentation_enabled": bool(args.augment_heuristic_splitting),
        "augmentation_parameters": {
            "n_aug_per_complete": args.n_aug_per_complete,
            "min_keep_tokens": args.min_keep_tokens,
        },
        "fold_macro_f1_default": fold_scores,
        "fold_macro_f1_default_mean": float(np.mean(fold_scores)),
        "fold_macro_f1_default_std": float(np.std(fold_scores)),
        "oof_macro_f1_default": float(oof_default),
        "oof_threshold": float(threshold),
        "oof_macro_f1_tuned": float(tuned_f1),
        "prediction_distribution_at_threshold": {str(k): int(v) for k, v in pred_dist.items()},
        "runtime_seconds": float(time.time() - start),
        "runtime_notes": "Final epoch probabilities; threshold tuned on OOF only; loaded model parameters cast to fp32 before training.",
        "device": torch.cuda.get_device_name(0),
        "oom_fallback_used": oom_fallbacks,
        "fold_diagnostics": fold_diagnostics,
    }
    write_json(output_dir / f"{args.model_tag}_meta.json", meta)
    maybe_write_ablation_report(output_dir)

    if args.model_tag.startswith("distilbert") and tuned_f1 < 0.90:
        raise SystemExit("DistilBERT tuned OOF Macro-F1 < 0.90; stop and inspect label alignment / augmentation.")


if __name__ == "__main__":
    main()
