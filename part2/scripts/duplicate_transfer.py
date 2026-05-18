from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pandas as pd

from part2_utils import ROOT, load_data, normalize_text, validate_submission, write_json


BASE_INPUT = ROOT / "submissions" / "submission_stacker_oof.csv"
FINAL_OUTPUT = ROOT / "submissions" / "submission_final.csv"


def main() -> None:
    train_df, test_df = load_data()
    if not BASE_INPUT.exists():
        raise FileNotFoundError(f"Missing base submission: {BASE_INPUT}")
    validate_submission(BASE_INPUT, test_df)
    base = pd.read_csv(BASE_INPUT)
    final = base.copy()

    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for _, row in train_df.iterrows():
        groups[normalize_text(row["text"], strip_trailing_punct=True)].append((str(row["id"]), int(row["label"])))

    audit_rows: list[dict] = []
    applied_indices: list[int] = []
    for idx, row in test_df.iterrows():
        norm = normalize_text(row["text"], strip_trailing_punct=True)
        matches = groups.get(norm, [])
        labels = [label for _, label in matches]
        ids = [match_id for match_id, _ in matches]
        base_label = int(base.iloc[idx]["label"])
        unanimous = len(set(labels)) == 1 if labels else False
        override_label = labels[0] if unanimous else None
        applied = False
        reason = "no_exact_train_match"
        if matches:
            reason = "diagnostic_only_count_lt_2"
        if matches and not unanimous:
            reason = "conflicting_train_labels"
        if matches and len(matches) >= 2 and unanimous:
            if override_label != base_label:
                final.loc[idx, "label"] = int(override_label)
                applied = True
                applied_indices.append(idx)
                reason = "safe_unanimous_duplicate_override"
            else:
                reason = "safe_unanimous_duplicate_already_same"
        audit_rows.append(
            {
                "test_row_index": int(idx),
                "test_id": row["id"],
                "original_text": row["text"],
                "normalized_text": norm,
                "base_label": base_label,
                "override_label": "" if override_label is None else int(override_label),
                "applied": applied,
                "train_match_count": len(matches),
                "train_match_ids": " ".join(ids),
                "train_match_labels": " ".join(str(label) for label in labels),
                "reason": reason,
            }
        )

    aborted = False
    if len(applied_indices) > 20:
        aborted = True
        final = base.copy()
        for audit in audit_rows:
            if audit["applied"]:
                audit["applied"] = False
                audit["reason"] = "aborted_override_count_gt_20"
        applied_indices = []

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_OUTPUT, index=False)
    validate_submission(FINAL_OUTPUT, test_df)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(ROOT / "outputs" / "duplicate_audit.csv", index=False)
    before_dist = base["label"].value_counts().sort_index().astype(int).to_dict()
    after_dist = final["label"].value_counts().sort_index().astype(int).to_dict()
    changed_ids = final.loc[base["label"].values != final["label"].values, "id"].astype(str).tolist()
    meta = {
        "base_input": str(BASE_INPUT),
        "final_output": str(FINAL_OUTPUT),
        "safe_override_rule": [
            "test normalized text exactly matches a train group",
            "train group count >= 2",
            "train group labels are unanimous",
            "override differs from current prediction",
        ],
        "aborted_due_to_override_guard": aborted,
        "applied_override_count": int(len(applied_indices)),
        "changed_ids": changed_ids,
        "distribution_before": {str(k): int(v) for k, v in before_dist.items()},
        "distribution_after": {str(k): int(v) for k, v in after_dist.items()},
        "diagnostic_exact_match_rows": int((audit["train_match_count"] >= 1).sum()),
        "conflicting_train_group_rows": int((audit["reason"] == "conflicting_train_labels").sum()),
    }
    write_json(ROOT / "outputs" / "duplicate_transfer_metadata.json", meta)
    print(f"duplicate overrides applied={len(applied_indices)} aborted={aborted}")
    print(f"distribution before={before_dist}")
    print(f"distribution after={after_dist}")
    print(f"changed ids={changed_ids}")
    print(f"wrote {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
