from __future__ import annotations

import re

import numpy as np
import pandas as pd

from part2_utils import ROOT, assert_finite, load_data, write_json


PREP = {"to", "of", "for", "with", "by", "in", "on", "at", "from", "into", "about", "like", "as"}
DET = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "his", "her", "their"}
CONJ = {"and", "or", "but", "because", "if", "when", "while", "so", "which", "who", "although", "though"}
AUX = {
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "shall",
    "must",
}
WH = {"what", "where", "when", "why", "how", "who", "which"}
REQUEST_PREFIXES = (
    "can you",
    "could you",
    "please",
    "tell me",
    "would you",
    "what is",
    "what are",
    "how do",
    "how to",
    "i need",
    "i want",
    "i would like",
)
FILLERS = ("uh", "um", "hmm", "like", "you know", "i mean", "actually", "wait", "so um", "well")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _contains_filler(lower: str) -> bool:
    for filler in FILLERS:
        if " " in filler:
            if filler in lower:
                return True
        elif re.search(rf"\b{re.escape(filler)}\b", lower):
            return True
    return False


def build_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "ends_with_prep",
        "ends_with_det",
        "ends_with_conj",
        "ends_with_aux",
        "ends_with_wh",
        "ends_with_comma",
        "ends_with_dash",
        "ends_with_ellipsis",
        "ends_with_question",
        "ends_with_period",
        "ends_with_exclaim",
        "no_terminal_punct",
        "starts_with_request",
        "has_filler",
        "word_count",
        "char_count",
        "log1p_word_count",
        "short_utterance",
        "medium_utterance",
        "long_utterance",
        "last_token_is_incomplete_group",
        "last_token_is_function_word",
        "last_token_len",
        "comma_count",
        "question_count",
        "terminal_punct_present",
    ]
    rows: list[list[float]] = []
    for text in texts:
        raw = "" if text is None else str(text)
        stripped = raw.strip()
        lower = stripped.lower()
        tokens = _tokens(lower)
        last = tokens[-1] if tokens else ""
        word_count = len(tokens)
        char_count = len(stripped)
        ends_with_ellipsis = lower.endswith("...") or lower.endswith("…")
        terminal_punct = stripped.endswith((".", "?", "!"))
        row = [
            float(last in PREP),
            float(last in DET),
            float(last in CONJ),
            float(last in AUX),
            float(last in WH),
            float(stripped.endswith(",")),
            float(stripped.endswith("-") or stripped.endswith("--") or stripped.endswith("—") or stripped.endswith("–")),
            float(ends_with_ellipsis),
            float(stripped.endswith("?")),
            float(stripped.endswith(".")),
            float(stripped.endswith("!")),
            float(not terminal_punct),
            float(any(lower.startswith(prefix) for prefix in REQUEST_PREFIXES)),
            float(_contains_filler(lower)),
            float(word_count),
            float(char_count),
            float(np.log1p(word_count)),
            float(word_count <= 3),
            float(4 <= word_count <= 8),
            float(word_count >= 15),
            float(last in PREP or last in DET or last in CONJ or last in AUX or last in WH),
            float(last in PREP or last in DET or last in CONJ or last in AUX),
            float(len(last)),
            float(stripped.count(",")),
            float(stripped.count("?")),
            float(terminal_punct),
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), feature_names


def main() -> None:
    train_df, test_df = load_data()
    train_features, feature_names = build_features(train_df["text"].tolist())
    test_features, _ = build_features(test_df["text"].tolist())
    assert_finite("endpoint_train", train_features)
    assert_finite("endpoint_test", test_features)

    y = train_df["label"].values.astype(float)
    rows = []
    for idx, name in enumerate(feature_names):
        values = train_features[:, idx].astype(float)
        if np.std(values) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, y)[0, 1])
        rows.append({"feature": name, "pearson_with_label1": corr, "mean": float(np.mean(values)), "std": float(np.std(values))})
    corr_df = pd.DataFrame(rows).sort_values("pearson_with_label1")

    out_dir = ROOT / "outputs" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "endpoint_train.npy", train_features)
    np.save(out_dir / "endpoint_test.npy", test_features)
    write_json(out_dir / "endpoint_feature_names.json", {"feature_names": feature_names})
    corr_df.to_csv(out_dir / "endpoint_correlations.csv", index=False)
    print(f"endpoint_train shape={train_features.shape}")
    print(f"endpoint_test shape={test_features.shape}")
    print("Endpoint feature correlations with label=1:")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
