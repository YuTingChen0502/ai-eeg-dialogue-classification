from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from part2_utils import ROOT, assert_finite, load_data, normalize_text, write_json


FEATURE_NAMES = [
    "nn_max_sim_word",
    "nn_max_sim_char",
    "top3_label1_mean_word",
    "top5_label1_mean_word",
    "top3_label1_mean_char",
    "top5_label1_mean_char",
    "top5_label_agreement_word",
    "top5_label_agreement_char",
    "exact_duplicate_flag",
    "exact_duplicate_label_if_any",
    "exact_duplicate_count",
    "near_duplicate_099_flag",
    "near_duplicate_099_label_if_any",
    "near_duplicate_099_similarity",
]


def _make_exact_groups(ref_df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for _, row in ref_df.iterrows():
        groups[normalize_text(row["text"], strip_trailing_punct=True)].append((str(row["id"]), int(row["label"])))
    return groups


def _top_indices(row, k: int) -> tuple[np.ndarray, np.ndarray]:
    arr = row.toarray().ravel()
    if arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    k = min(k, arr.size)
    if k == arr.size:
        idx = np.argsort(-arr)
    else:
        idx = np.argpartition(-arr, np.arange(k))[:k]
        idx = idx[np.argsort(-arr[idx])]
    return idx[:k], arr[idx[:k]]


def _label_means(labels: np.ndarray, idx: np.ndarray, k: int) -> float:
    if len(idx) == 0:
        return 0.5
    take = idx[: min(k, len(idx))]
    return float(np.mean(labels[take]))


def _agreement(labels: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return 0.0
    mean = float(np.mean(labels[idx[: min(5, len(idx))]]))
    return max(mean, 1.0 - mean)


def _duplicate_values(query_text: str, exact_groups: dict[str, list[tuple[str, int]]]) -> tuple[float, float, float, list[str], list[int]]:
    norm = normalize_text(query_text, strip_trailing_punct=True)
    matches = exact_groups.get(norm, [])
    labels = [label for _, label in matches]
    ids = [match_id for match_id, _ in matches]
    if not matches:
        return 0.0, -1.0, 0.0, [], []
    label_value = float(labels[0]) if len(set(labels)) == 1 else -1.0
    return 1.0, label_value, float(len(matches)), ids, labels


def _build_block(query_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    labels = ref_df["label"].values.astype(int)
    ref_texts = ref_df["text"].tolist()
    query_texts = query_df["text"].tolist()
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    word_ref = word_vec.fit_transform(ref_texts)
    word_query = word_vec.transform(query_texts)
    char_ref = char_vec.fit_transform(ref_texts)
    char_query = char_vec.transform(query_texts)
    word_sim = word_query @ word_ref.T
    char_sim = char_query @ char_ref.T
    exact_groups = _make_exact_groups(ref_df)

    features = np.zeros((len(query_df), len(FEATURE_NAMES)), dtype=np.float32)
    audits: list[dict] = []
    for i, (_, query_row) in enumerate(query_df.iterrows()):
        word_idx, word_vals = _top_indices(word_sim.getrow(i), 5)
        char_idx, char_vals = _top_indices(char_sim.getrow(i), 5)
        exact_flag, exact_label, exact_count, exact_ids, exact_labels = _duplicate_values(query_row["text"], exact_groups)
        near_flag = 0.0
        near_label = -1.0
        near_similarity = 0.0
        near_id = ""
        if len(char_idx) > 0:
            cand_idx = int(char_idx[0])
            near_similarity = float(SequenceMatcher(None, normalize_text(query_row["text"]), normalize_text(ref_texts[cand_idx])).ratio())
            if near_similarity >= 0.99:
                near_flag = 1.0
                near_label = float(labels[cand_idx])
                near_id = str(ref_df.iloc[cand_idx]["id"])

        features[i, :] = [
            float(word_vals[0]) if len(word_vals) else 0.0,
            float(char_vals[0]) if len(char_vals) else 0.0,
            _label_means(labels, word_idx, 3),
            _label_means(labels, word_idx, 5),
            _label_means(labels, char_idx, 3),
            _label_means(labels, char_idx, 5),
            _agreement(labels, word_idx),
            _agreement(labels, char_idx),
            exact_flag,
            exact_label,
            exact_count,
            near_flag,
            near_label,
            near_similarity,
        ]
        audits.append(
            {
                "row_index": int(query_row.name),
                "id": query_row["id"],
                "normalized_text": normalize_text(query_row["text"], strip_trailing_punct=True),
                "exact_duplicate_flag": int(exact_flag),
                "exact_duplicate_label_if_any": int(exact_label),
                "exact_duplicate_count": int(exact_count),
                "exact_duplicate_ids": " ".join(exact_ids),
                "exact_duplicate_labels": " ".join(str(x) for x in exact_labels),
                "near_duplicate_099_flag": int(near_flag),
                "near_duplicate_099_label_if_any": int(near_label),
                "near_duplicate_099_similarity": near_similarity,
                "near_duplicate_id": near_id,
            }
        )
    return features, audits


def main() -> None:
    start = time.time()
    train_df, test_df = load_data()
    y = train_df["label"].values.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_features = np.zeros((len(train_df), len(FEATURE_NAMES)), dtype=np.float32)
    audits: list[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], y)):
        print(f"retrieval fold {fold_idx + 1}")
        ref_df = train_df.iloc[tr_idx].reset_index(drop=True)
        query_df = train_df.iloc[va_idx].copy()
        block, block_audit = _build_block(query_df, ref_df)
        train_features[va_idx, :] = block
        for row in block_audit:
            row["split"] = "train_oof"
            row["fold"] = fold_idx + 1
        audits.extend(block_audit)

    print("retrieval test against full train")
    test_query_df = test_df.copy()
    test_features, test_audit = _build_block(test_query_df, train_df.reset_index(drop=True))
    for row in test_audit:
        row["split"] = "test"
        row["fold"] = 0
    audits.extend(test_audit)
    assert_finite("retrieval_train", train_features)
    assert_finite("retrieval_test", test_features)

    out_dir = ROOT / "outputs" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "retrieval_train.npy", train_features)
    np.save(out_dir / "retrieval_test.npy", test_features)
    write_json(out_dir / "retrieval_feature_names.json", {"feature_names": FEATURE_NAMES})
    pd.DataFrame(audits).to_csv(out_dir / "retrieval_audit.csv", index=False)
    meta = {
        "shape_train": list(train_features.shape),
        "shape_test": list(test_features.shape),
        "feature_names": FEATURE_NAMES,
        "train_retrieval": "out-of-fold; each validation row retrieves only from that fold's training rows",
        "test_retrieval": "full train retrieval; labels used only as stacker features",
        "runtime_seconds": float(time.time() - start),
        "train_exact_duplicate_rows": int(np.sum(train_features[:, FEATURE_NAMES.index("exact_duplicate_flag")] > 0)),
        "test_exact_duplicate_rows": int(np.sum(test_features[:, FEATURE_NAMES.index("exact_duplicate_flag")] > 0)),
        "test_near_duplicate_099_rows": int(np.sum(test_features[:, FEATURE_NAMES.index("near_duplicate_099_flag")] > 0)),
    }
    write_json(out_dir / "retrieval_meta.json", meta)
    print(f"retrieval_train shape={train_features.shape}")
    print(f"retrieval_test shape={test_features.shape}")
    print(f"test exact duplicate rows={meta['test_exact_duplicate_rows']}")
    print(f"test near duplicate >=0.99 rows={meta['test_near_duplicate_099_rows']}")


if __name__ == "__main__":
    main()
