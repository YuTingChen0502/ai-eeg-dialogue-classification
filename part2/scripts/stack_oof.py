from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from part2_utils import (
    BASELINE_SUBMISSION,
    ROOT,
    assert_finite,
    labels_from_probs,
    load_data,
    read_meta,
    tune_threshold,
    validate_submission,
    write_json,
    write_submission,
)


def _load_feature_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["feature_names"])


def _choose_distilbert_tag(oof_dir: Path) -> tuple[str | None, str]:
    aug = oof_dir / "distilbert_meta.json"
    no_aug = oof_dir / "distilbert_no_aug_meta.json"
    if aug.exists() and no_aug.exists():
        aug_meta = read_meta(aug)
        no_aug_meta = read_meta(no_aug)
        delta = float(aug_meta["oof_macro_f1_tuned"]) - float(no_aug_meta["oof_macro_f1_tuned"])
        if delta < -0.005:
            return "distilbert_no_aug", "DistilBERT ablation selected no-augmentation version."
        return "distilbert", "DistilBERT ablation kept heuristic splitting version."
    if aug.exists():
        return "distilbert", "Only augmented DistilBERT metadata found."
    if no_aug.exists():
        return "distilbert_no_aug", "Only no-augmentation DistilBERT metadata found."
    return None, "No DistilBERT OOF files found."


def _load_transformers(train_len: int, test_len: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[dict]]:
    oof_dir = ROOT / "outputs" / "oof"
    selected: list[str] = []
    distilbert_tag, distilbert_note = _choose_distilbert_tag(oof_dir)
    if distilbert_tag is not None:
        selected.append(distilbert_tag)
    for tag in ["deberta_v3", "roberta"]:
        if (oof_dir / f"{tag}_meta.json").exists():
            selected.append(tag)

    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []
    notes = [distilbert_note]
    metas: list[dict] = []
    for tag in selected:
        train_path = oof_dir / f"{tag}_oof.npy"
        test_path = oof_dir / f"{tag}_test.npy"
        meta_path = oof_dir / f"{tag}_meta.json"
        if not train_path.exists() or not test_path.exists() or not meta_path.exists():
            notes.append(f"Skipped {tag}: missing OOF/test/meta file.")
            continue
        train_arr = np.load(train_path).reshape(-1, 1)
        test_arr = np.load(test_path).reshape(-1, 1)
        if train_arr.shape[0] != train_len or test_arr.shape[0] != test_len:
            raise ValueError(f"{tag} shape mismatch: train {train_arr.shape}, test {test_arr.shape}")
        assert_finite(f"{tag}_oof", train_arr)
        assert_finite(f"{tag}_test", test_arr)
        train_blocks.append(train_arr.astype(np.float32))
        test_blocks.append(test_arr.astype(np.float32))
        names.append(f"transformer:{tag}")
        meta = read_meta(meta_path)
        meta["tag"] = tag
        metas.append(meta)
    return train_blocks, test_blocks, names, notes, metas


def _append_if_exists(
    train_blocks: list[np.ndarray],
    test_blocks: list[np.ndarray],
    feature_names: list[str],
    scale_cols: list[int],
    train_path: Path,
    test_path: Path,
    names_path: Path,
    prefix: str,
    train_len: int,
    test_len: int,
    notes: list[str],
) -> None:
    if not train_path.exists() or not test_path.exists() or not names_path.exists():
        notes.append(f"Skipped {prefix}: missing feature files.")
        return
    train_arr = np.load(train_path)
    test_arr = np.load(test_path)
    names = _load_feature_names(names_path)
    if train_arr.shape[0] != train_len or test_arr.shape[0] != test_len:
        raise ValueError(f"{prefix} shape mismatch: train {train_arr.shape}, test {test_arr.shape}")
    if train_arr.shape[1] != len(names) or test_arr.shape[1] != len(names):
        raise ValueError(f"{prefix} feature name count mismatch")
    assert_finite(f"{prefix}_train", train_arr)
    assert_finite(f"{prefix}_test", test_arr)
    start_col = len(feature_names)
    train_blocks.append(train_arr.astype(np.float32))
    test_blocks.append(test_arr.astype(np.float32))
    feature_names.extend([f"{prefix}:{name}" for name in names])
    scale_cols.extend(range(start_col, start_col + train_arr.shape[1]))
    notes.append(f"Loaded {prefix} features with shape {train_arr.shape}.")


def _scaled(train_x: np.ndarray, valid_x: np.ndarray, scale_cols: list[int]) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    if not scale_cols:
        return train_x.copy(), valid_x.copy(), None
    scaler = StandardScaler()
    train_out = train_x.copy()
    valid_out = valid_x.copy()
    train_out[:, scale_cols] = scaler.fit_transform(train_out[:, scale_cols])
    valid_out[:, scale_cols] = scaler.transform(valid_out[:, scale_cols])
    return train_out, valid_out, scaler


def main() -> None:
    start = time.time()
    train_df, test_df = load_data()
    y = train_df["label"].values.astype(int)
    train_blocks, test_blocks, feature_names, notes, transformer_metas = _load_transformers(len(train_df), len(test_df))
    scale_cols: list[int] = []
    feature_dir = ROOT / "outputs" / "features"

    _append_if_exists(
        train_blocks,
        test_blocks,
        feature_names,
        scale_cols,
        feature_dir / "classical_train.npy",
        feature_dir / "classical_test.npy",
        feature_dir / "classical_feature_names.json",
        "classical",
        len(train_df),
        len(test_df),
        notes,
    )
    _append_if_exists(
        train_blocks,
        test_blocks,
        feature_names,
        scale_cols,
        feature_dir / "endpoint_train.npy",
        feature_dir / "endpoint_test.npy",
        feature_dir / "endpoint_feature_names.json",
        "endpoint",
        len(train_df),
        len(test_df),
        notes,
    )
    _append_if_exists(
        train_blocks,
        test_blocks,
        feature_names,
        scale_cols,
        feature_dir / "retrieval_train.npy",
        feature_dir / "retrieval_test.npy",
        feature_dir / "retrieval_feature_names.json",
        "retrieval",
        len(train_df),
        len(test_df),
        notes,
    )

    if not train_blocks:
        raise SystemExit("No usable feature blocks found for stacker.")
    x_train = np.hstack(train_blocks).astype(np.float32)
    x_test = np.hstack(test_blocks).astype(np.float32)
    assert_finite("stacker_x_train", x_train)
    assert_finite("stacker_x_test", x_test)
    print(f"stacker X_train shape={x_train.shape}")
    print(f"stacker X_test shape={x_test.shape}")
    print(f"features used={feature_names}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_values = [0.3, 1.0, 3.0]
    candidates: list[dict] = []
    for c_value in c_values:
        oof_probs = np.zeros(len(train_df), dtype=np.float32)
        fold_f1s: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(x_train, y)):
            x_tr, x_va, _ = _scaled(x_train[tr_idx], x_train[va_idx], scale_cols)
            clf = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=42)
            clf.fit(x_tr, y[tr_idx])
            probs = clf.predict_proba(x_va)[:, 1]
            oof_probs[va_idx] = probs.astype(np.float32)
            fold_f1 = f1_score(y[va_idx], labels_from_probs(probs, 0.5), average="macro")
            fold_f1s.append(float(fold_f1))
        threshold, tuned_f1 = tune_threshold(y, oof_probs, 0.30, 0.70, 0.01)
        default_f1 = f1_score(y, labels_from_probs(oof_probs, 0.5), average="macro")
        candidates.append(
            {
                "C": c_value,
                "oof_probs": oof_probs,
                "threshold": float(threshold),
                "tuned_f1": float(tuned_f1),
                "default_f1": float(default_f1),
                "fold_macro_f1_default": fold_f1s,
            }
        )
        print(f"stacker C={c_value}: Macro-F1@0.5={default_f1:.4f}, tuned={tuned_f1:.4f}, threshold={threshold:.2f}")
    best = max(candidates, key=lambda item: item["tuned_f1"])

    x_full = x_train.copy()
    x_test_scaled = x_test.copy()
    final_scaler = None
    if scale_cols:
        final_scaler = StandardScaler()
        x_full[:, scale_cols] = final_scaler.fit_transform(x_full[:, scale_cols])
        x_test_scaled[:, scale_cols] = final_scaler.transform(x_test_scaled[:, scale_cols])
    final_clf = LogisticRegression(C=best["C"], max_iter=2000, solver="lbfgs", random_state=42)
    final_clf.fit(x_full, y)
    stacker_test_probs = final_clf.predict_proba(x_test_scaled)[:, 1].astype(np.float32)
    stacker_oof_probs = best["oof_probs"].astype(np.float32)
    assert_finite("stacker_oof_probs", stacker_oof_probs)
    assert_finite("stacker_test_probs", stacker_test_probs)
    np.save(ROOT / "outputs" / "stacker_oof_probs.npy", stacker_oof_probs)
    np.save(ROOT / "outputs" / "stacker_test_probs.npy", stacker_test_probs)

    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": final_clf.coef_[0],
            "scaled": [idx in scale_cols for idx in range(len(feature_names))],
        }
    )
    feature_table["abs_coefficient"] = feature_table["coefficient"].abs()
    feature_table.sort_values("abs_coefficient", ascending=False).to_csv(ROOT / "outputs" / "stacker_feature_table.csv", index=False)

    if transformer_metas:
        best_single = max(transformer_metas, key=lambda meta: float(meta["oof_macro_f1_tuned"]))
    else:
        best_single = None
    fallback_used = False
    chosen_threshold = float(best["threshold"])
    chosen_probs = stacker_test_probs
    chosen_oof_f1 = float(best["tuned_f1"])
    chosen_source = "stacker"
    if best_single is not None and chosen_oof_f1 < float(best_single["oof_macro_f1_tuned"]):
        fallback_used = True
        chosen_source = f"fallback:{best_single['tag']}"
        chosen_threshold = float(best_single["oof_threshold"])
        chosen_probs = np.load(ROOT / "outputs" / "oof" / f"{best_single['tag']}_test.npy")
        chosen_oof_f1 = float(best_single["oof_macro_f1_tuned"])

    labels = labels_from_probs(chosen_probs, chosen_threshold)
    sub_path = ROOT / "submissions" / "submission_stacker_oof.csv"
    write_submission(sub_path, test_df, labels)
    validation = validate_submission(sub_path, test_df)
    baseline = pd.read_csv(ROOT / BASELINE_SUBMISSION)
    hamming = int(np.sum(baseline["label"].values != labels))
    distribution = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()

    meta = {
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "feature_names": feature_names,
        "scale_columns": scale_cols,
        "notes": notes,
        "candidates": [
            {k: v for k, v in cand.items() if k != "oof_probs"}
            for cand in candidates
        ],
        "selected_C": float(best["C"]),
        "stacker_oof_macro_f1_default": float(best["default_f1"]),
        "stacker_oof_threshold": float(best["threshold"]),
        "stacker_oof_macro_f1_tuned": float(best["tuned_f1"]),
        "best_single_model": None
        if best_single is None
        else {
            "tag": best_single["tag"],
            "oof_macro_f1_tuned": float(best_single["oof_macro_f1_tuned"]),
            "threshold": float(best_single["oof_threshold"]),
        },
        "fallback_used": fallback_used,
        "chosen_source": chosen_source,
        "chosen_threshold": chosen_threshold,
        "chosen_oof_macro_f1": chosen_oof_f1,
        "submission_validation": validation,
        "prediction_distribution": {str(k): int(v) for k, v in distribution.items()},
        "hamming_vs_rollback_baseline": hamming,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(ROOT / "outputs" / "stacker_metadata.json", meta)
    print(f"stacker tuned OOF Macro-F1={best['tuned_f1']:.4f} threshold={best['threshold']:.2f}")
    print(f"fallback_used={fallback_used} chosen_source={chosen_source}")
    print(f"final distribution={distribution}")
    print(f"Hamming distance vs {BASELINE_SUBMISSION}: {hamming}")
    print(f"wrote {sub_path}")


if __name__ == "__main__":
    main()
