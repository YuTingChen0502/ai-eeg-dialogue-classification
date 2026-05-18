from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from part2_utils import ROOT, load_data, standardize_frame


def augment_with_splits(
    df: pd.DataFrame,
    rng_seed: int = 42,
    n_aug_per_complete: int = 1,
    min_keep_tokens: int = 3,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    For each row with label == 1 and len(tokens) >= 5, generate truncated prefixes
    as label 0 incomplete samples.
    Return original rows plus augmented rows.
    Do not modify original rows.
    Augmented ids are unique and non-colliding: aug_{original_id}_{cut}_{j}.
    """
    if "label" not in df.columns:
        raise KeyError("augment_with_splits requires a label column")
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    original = df.copy(deep=True)
    rng = np.random.default_rng(rng_seed)
    existing_texts = set(original[text_col].fillna("").astype(str).tolist())
    existing_ids = set(original["id"].astype(str).tolist()) if "id" in original.columns else set()
    augmented_rows: list[dict] = []

    for _, row in original.iterrows():
        if int(row["label"]) != 1:
            continue
        tokens = str(row[text_col]).split()
        if len(tokens) < 5:
            continue
        source_id = row["id"] if "id" in original.columns else int(row.name)
        for j in range(n_aug_per_complete):
            cut = int(rng.integers(min_keep_tokens, len(tokens)))
            if cut >= len(tokens):
                continue
            aug_text = " ".join(tokens[:cut])
            if aug_text in existing_texts:
                continue
            aug_id = f"aug_{source_id}_{cut}_{j}"
            suffix = 1
            while aug_id in existing_ids:
                aug_id = f"aug_{source_id}_{cut}_{j}_{suffix}"
                suffix += 1
            new_row = row.to_dict()
            new_row[text_col] = aug_text
            new_row["label"] = 0
            new_row["id"] = aug_id
            augmented_rows.append(new_row)
            existing_texts.add(aug_text)
            existing_ids.add(aug_id)

    if not augmented_rows:
        return original.reset_index(drop=True)
    out = pd.concat([original, pd.DataFrame(augmented_rows)], ignore_index=True)
    return out


def _sanity() -> None:
    train_df, _ = load_data()
    train_df = standardize_frame(train_df)
    augmented = augment_with_splits(train_df, rng_seed=42)
    augmented2 = augment_with_splits(train_df, rng_seed=42)
    if not augmented.equals(augmented2):
        raise AssertionError("augmentation is not deterministic for the same seed")
    if augmented["id"].duplicated().any():
        raise AssertionError("duplicate ids found after augmentation")
    aug_only = augmented[augmented["id"].astype(str).str.startswith("aug_")].copy()
    if not aug_only.empty and not (aug_only["label"] == 0).all():
        raise AssertionError("all augmented rows must have label 0")
    source_text_by_id = train_df.set_index("id")["text"].astype(str).to_dict()
    for _, row in aug_only.iterrows():
        parts = str(row["id"]).split("_")
        source_id = "_".join(parts[1:-2]) if len(parts) > 4 else parts[1]
        if source_id in source_text_by_id and len(str(row["text"]).split()) >= len(source_text_by_id[source_id].split()):
            raise AssertionError("augmented row is not shorter than source")

    print(f"original size: {len(train_df)}")
    print(f"augmented size: {len(augmented)}")
    print("label distribution before:")
    print(train_df["label"].value_counts().sort_index().to_string())
    print("label distribution after:")
    print(augmented["label"].value_counts().sort_index().to_string())
    print("sample augmented rows:")
    cols = ["id", "text", "label"]
    print(aug_only[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    _sanity()
