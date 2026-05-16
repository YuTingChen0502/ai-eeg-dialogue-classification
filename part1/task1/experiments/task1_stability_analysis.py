"""Internal-only Task 1 stability analysis.

This script does not write Kaggle submission files. It analyzes whether stable
regularized CSP + shrinkage-LDA models agree on the Task 1 test rows.

Run from the repository root:
    python part1/task1/experiments/task1_stability_analysis.py

Or from part1/task1:
    python experiments/task1_stability_analysis.py
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import robust_task1_experiment as robust


SEED = 42
CLASSES = robust.CLASSES
REQUIRED_BANDS = {
    "alpha_mu_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "broad_4_40": (4.0, 40.0),
    "high_gamma_70_125": (70.0, 125.0),
}
REFERENCE_FILES = [
    "task1_submission_ensemble_best.csv",
    "task1_submission_multiband_balanced.csv",
    "task1_submission_hybrid_two_public_winners.csv",
]


@dataclass(frozen=True)
class StabilityConfig:
    band_name: str
    band: tuple[float, float]
    preprocess: str
    csp_components: int
    csp_shrinkage: float

    @property
    def key(self) -> tuple:
        return (
            self.band_name,
            self.preprocess,
            self.csp_components,
            round(self.csp_shrinkage, 6),
        )


def find_data_dir() -> Path:
    return robust.find_data_dir()


def load_data() -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data_dir = find_data_dir()
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    test_npz = np.load(data_dir / "test.npz", allow_pickle=True)
    x_train = np.asarray(train_npz["x"], dtype=np.float64)
    y_train = np.asarray(train_npz["y"], dtype=int)
    x_test = np.asarray(test_npz["x"], dtype=np.float64)
    test_ids = np.asarray(test_npz["id"], dtype=int) if "id" in test_npz.files else np.arange(len(x_test), dtype=int)
    sfreq = robust.read_scalar(train_npz, "sfreq", 250.0)
    return data_dir, x_train, y_train, x_test, test_ids, sfreq


def build_configs() -> list[StabilityConfig]:
    configs = []
    for band_name, band in REQUIRED_BANDS.items():
        for preprocess in ("demean", "car"):
            for csp_components in (2, 4):
                for csp_shrinkage in (0.15, 0.25, 0.35):
                    configs.append(
                        StabilityConfig(
                            band_name=band_name,
                            band=band,
                            preprocess=preprocess,
                            csp_components=csp_components,
                            csp_shrinkage=csp_shrinkage,
                        )
                    )
    return configs


def precompute_filtered(
    x_train: np.ndarray,
    x_test: np.ndarray,
    sfreq: float,
    configs: list[StabilityConfig],
) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    filtered_train = {}
    filtered_test = {}
    for preprocess in sorted({cfg.preprocess for cfg in configs}):
        train_pre = robust.preprocess_trials(x_train, preprocess)
        test_pre = robust.preprocess_trials(x_test, preprocess)
        for band in sorted({cfg.band for cfg in configs}):
            filtered_train[(preprocess, band)] = robust.bandpass(train_pre, band, sfreq)
            filtered_test[(preprocess, band)] = robust.bandpass(test_pre, band, sfreq)
    return filtered_train, filtered_test


def fit_predict_scores(
    cfg: StabilityConfig,
    x_fit_band: np.ndarray,
    y_fit: np.ndarray,
    x_eval_band: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    csp = robust.RegularizedOVRCSP(cfg.csp_components, cfg.csp_shrinkage).fit(x_fit_band, y_fit)
    f_fit = csp.transform(x_fit_band)
    f_eval = csp.transform(x_eval_band)
    scaler = StandardScaler().fit(f_fit)
    f_fit_s = scaler.transform(f_fit)
    f_eval_s = scaler.transform(f_eval)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(f_fit_s, y_fit)
    pred = clf.predict(f_eval_s).astype(int)
    scores = robust.aligned_scores(clf, f_fit_s, f_eval_s)
    return pred, scores


def evaluate_config(
    cfg: StabilityConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    filtered_train: dict[tuple, np.ndarray],
) -> dict:
    repeated = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=SEED)
    loo = LeaveOneOut()
    splitters = {
        "rskf20": list(repeated.split(x_train, y_train)),
        "loo": list(loo.split(x_train, y_train)),
    }
    metrics = {}
    x_band = filtered_train[(cfg.preprocess, cfg.band)]
    for split_name, splits in splitters.items():
        fold_f1 = []
        fold_acc = []
        all_true = []
        all_pred = []
        for tr_idx, va_idx in splits:
            pred, _ = fit_predict_scores(cfg, x_band[tr_idx], y_train[tr_idx], x_band[va_idx])
            y_eval = y_train[va_idx]
            fold_f1.append(f1_score(y_eval, pred, average="macro", labels=CLASSES, zero_division=0))
            fold_acc.append(accuracy_score(y_eval, pred))
            all_true.extend(y_eval.tolist())
            all_pred.extend(pred.tolist())
        metrics[split_name] = {
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "fold_accuracy_mean": float(np.mean(fold_acc)),
            "fold_accuracy_std": float(np.std(fold_acc)),
            "aggregate_macro_f1": float(
                f1_score(all_true, all_pred, average="macro", labels=CLASSES, zero_division=0)
            ),
            "aggregate_accuracy": float(accuracy_score(all_true, all_pred)),
        }
    return metrics


def selection_score(metrics: dict) -> float:
    return float(
        metrics["rskf20"]["fold_macro_f1_mean"]
        - 0.25 * metrics["rskf20"]["fold_macro_f1_std"]
        + 0.20 * metrics["loo"]["aggregate_macro_f1"]
    )


def fit_full(
    cfg: StabilityConfig,
    y_train: np.ndarray,
    filtered_train: dict[tuple, np.ndarray],
    filtered_test: dict[tuple, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_train_band = filtered_train[(cfg.preprocess, cfg.band)]
    x_test_band = filtered_test[(cfg.preprocess, cfg.band)]
    return fit_predict_scores(cfg, x_train_band, y_train, x_test_band)


def prediction_distribution(pred: np.ndarray) -> dict[str, int]:
    return robust.prediction_distribution(np.asarray(pred, dtype=int))


def entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log2(nz)) / math.log2(len(CLASSES)))


def load_references(output_dir: Path, test_ids: np.ndarray) -> tuple[dict[str, np.ndarray], list[str]]:
    references = {}
    missing = []
    for name in REFERENCE_FILES:
        path = output_dir / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        if list(df.columns) != ["id", "label"] or df["id"].tolist() != test_ids.tolist():
            missing.append(f"{name} (invalid format/order)")
            continue
        references[name] = df["label"].to_numpy(dtype=int)
    return references, missing


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    rows = []
    columns = [str(col) for col in df.columns]
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    np.random.seed(SEED)

    data_dir, x_train, y_train, x_test, test_ids, sfreq = load_data()
    output_dir = data_dir.parent
    configs = build_configs()
    filtered_train, filtered_test = precompute_filtered(x_train, x_test, sfreq, configs)
    references, missing_refs = load_references(output_dir, test_ids)

    print(f"Data dir: {data_dir.resolve()}")
    print(f"x_train={x_train.shape}, x_test={x_test.shape}, sfreq={sfreq}")
    print(f"Stability configs: {len(configs)}")
    print(f"References found: {sorted(references)}")
    print(f"References missing: {missing_refs}")

    records = []
    full_predictions = {}
    full_scores = {}
    for i, cfg in enumerate(configs, start=1):
        metrics = evaluate_config(cfg, x_train, y_train, filtered_train)
        pred, scores = fit_full(cfg, y_train, filtered_train, filtered_test)
        full_predictions[cfg.key] = pred
        full_scores[cfg.key] = scores
        records.append(
            {
                **asdict(cfg),
                "band": list(cfg.band),
                "feature_dim": int(len(CLASSES) * cfg.csp_components),
                "rskf20_macro_f1_mean": metrics["rskf20"]["fold_macro_f1_mean"],
                "rskf20_macro_f1_std": metrics["rskf20"]["fold_macro_f1_std"],
                "rskf20_accuracy_mean": metrics["rskf20"]["fold_accuracy_mean"],
                "loo_accuracy": metrics["loo"]["aggregate_accuracy"],
                "loo_macro_f1": metrics["loo"]["aggregate_macro_f1"],
                "selection_score": selection_score(metrics),
                "test_distribution": json.dumps(prediction_distribution(pred), sort_keys=True),
            }
        )
        print(
            f"{i:02d}/{len(configs)} {cfg.band_name} {cfg.preprocess} "
            f"csp={cfg.csp_components} shrink={cfg.csp_shrinkage:.2f} "
            f"rskf={records[-1]['rskf20_macro_f1_mean']:.3f}+/-{records[-1]['rskf20_macro_f1_std']:.3f} "
            f"loo={records[-1]['loo_macro_f1']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(records).sort_values(
        ["selection_score", "rskf20_macro_f1_mean", "loo_macro_f1"],
        ascending=[False, False, False],
    )
    results_path = output_dir / "task1_stability_model_results.csv"
    results.to_csv(results_path, index=False)

    per_band = (
        results.groupby("band_name")
        .agg(
            top_selection_score=("selection_score", "max"),
            mean_selection_score=("selection_score", "mean"),
            top_rskf20_macro_f1=("rskf20_macro_f1_mean", "max"),
            top_loo_macro_f1=("loo_macro_f1", "max"),
            configs=("band_name", "count"),
        )
        .sort_values("top_selection_score", ascending=False)
        .reset_index()
    )
    per_band_path = output_dir / "task1_stability_band_summary.csv"
    per_band.to_csv(per_band_path, index=False)

    selected_rows = []
    selected_keys = set()
    for band_name in REQUIRED_BANDS:
        band_rows = results[results["band_name"] == band_name].head(2)
        for _, row in band_rows.iterrows():
            key = (
                row["band_name"],
                row["preprocess"],
                int(row["csp_components"]),
                round(float(row["csp_shrinkage"]), 6),
            )
            if key not in selected_keys:
                selected_rows.append(row.to_dict())
                selected_keys.add(key)
    for _, row in results.head(12).iterrows():
        key = (
            row["band_name"],
            row["preprocess"],
            int(row["csp_components"]),
            round(float(row["csp_shrinkage"]), 6),
        )
        if key not in selected_keys:
            selected_rows.append(row.to_dict())
            selected_keys.add(key)
        if len(selected_rows) >= 14:
            break

    selected_predictions = []
    selected_scores = []
    for row in selected_rows:
        key = (
            row["band_name"],
            row["preprocess"],
            int(row["csp_components"]),
            round(float(row["csp_shrinkage"]), 6),
        )
        selected_predictions.append(full_predictions[key])
        selected_scores.append(full_scores[key])

    pred_matrix = np.vstack(selected_predictions)
    mean_scores = np.mean(selected_scores, axis=0)
    consensus_pred = CLASSES[np.argmax(mean_scores, axis=1)]
    row_records = []
    for col_idx, sample_id in enumerate(test_ids.tolist()):
        votes = np.bincount(pred_matrix[:, col_idx], minlength=len(CLASSES))
        sorted_votes = np.sort(votes)[::-1]
        vote_fraction = float(sorted_votes[0] / len(selected_rows))
        vote_margin = float((sorted_votes[0] - sorted_votes[1]) / len(selected_rows))
        score_order = np.sort(mean_scores[col_idx])[::-1]
        score_margin = float(score_order[0] - score_order[1])
        if vote_fraction >= 0.75:
            stability = "stable"
        elif vote_fraction >= 0.55:
            stability = "mixed"
        else:
            stability = "unstable"
        row = {
            "id": int(sample_id),
            "consensus_label": int(consensus_pred[col_idx]),
            "stability": stability,
            "vote_fraction": vote_fraction,
            "vote_margin": vote_margin,
            "score_margin": score_margin,
            "vote_entropy": entropy_from_counts(votes),
            "votes_0": int(votes[0]),
            "votes_1": int(votes[1]),
            "votes_2": int(votes[2]),
            "votes_3": int(votes[3]),
        }
        for ref_name, ref_pred in references.items():
            short = ref_name.replace("task1_submission_", "").replace(".csv", "")
            row[f"{short}_label"] = int(ref_pred[col_idx])
            row[f"matches_{short}"] = bool(consensus_pred[col_idx] == ref_pred[col_idx])
        row_records.append(row)

    rows_df = pd.DataFrame(row_records)
    rows_path = output_dir / "task1_stability_test_rows.csv"
    rows_df.to_csv(rows_path, index=False)

    hamming_summary = {}
    for ref_name, ref_pred in references.items():
        hamming_summary[ref_name] = {
            "hamming_vs_consensus": robust.hamming(consensus_pred, ref_pred),
            "changed_ids_vs_consensus": robust.changed_row_ids(test_ids, consensus_pred, ref_pred),
            "reference_distribution": prediction_distribution(ref_pred),
        }

    consensus_summary = {
        "selected_model_count": len(selected_rows),
        "selected_models": [
            {
                "band_name": row["band_name"],
                "preprocess": row["preprocess"],
                "csp_components": int(row["csp_components"]),
                "csp_shrinkage": float(row["csp_shrinkage"]),
                "selection_score": float(row["selection_score"]),
                "rskf20_macro_f1_mean": float(row["rskf20_macro_f1_mean"]),
                "rskf20_macro_f1_std": float(row["rskf20_macro_f1_std"]),
                "loo_macro_f1": float(row["loo_macro_f1"]),
            }
            for row in selected_rows
        ],
        "consensus_distribution": prediction_distribution(consensus_pred),
        "stable_rows": rows_df[rows_df["stability"] == "stable"]["id"].astype(int).tolist(),
        "mixed_rows": rows_df[rows_df["stability"] == "mixed"]["id"].astype(int).tolist(),
        "unstable_rows": rows_df[rows_df["stability"] == "unstable"]["id"].astype(int).tolist(),
        "hamming_summary": hamming_summary,
        "references_found": sorted(references.keys()),
        "references_missing": missing_refs,
        "recommendation": {
            "kaggle_action": "do_not_submit",
            "preferred_files": [
                "task1_submission_ensemble_best.csv",
                "task1_submission_multiband_balanced.csv",
            ],
            "rationale": (
                "The internal consensus remains model-uncertain on several rows and does not overcome "
                "the known 0.625 public hedge; no public probing or row flipping is justified."
            ),
        },
    }
    summary_path = output_dir / "task1_stability_summary.json"
    summary_path.write_text(json.dumps(consensus_summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Task 1 Internal Stability Analysis",
        "",
        "This is an internal diagnostic only. It does not create Kaggle candidate CSVs.",
        "",
        "## Band Summary",
        "",
        dataframe_to_markdown(per_band, floatfmt=".4f"),
        "",
        "## Consensus Rows",
        "",
        dataframe_to_markdown(rows_df, floatfmt=".3f"),
        "",
        "## Recommendation",
        "",
        consensus_summary["recommendation"]["rationale"],
    ]
    report_path = output_dir / "task1_stability_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nBand summary:")
    print(per_band.to_string(index=False))
    print("\nConsensus distribution:", consensus_summary["consensus_distribution"])
    print("Stable rows:", consensus_summary["stable_rows"])
    print("Mixed rows:", consensus_summary["mixed_rows"])
    print("Unstable rows:", consensus_summary["unstable_rows"])
    print("Hamming summary:", json.dumps(hamming_summary, indent=2))
    print(f"\nWrote diagnostics: {results_path.name}, {per_band_path.name}, {rows_path.name}, {summary_path.name}, {report_path.name}")


if __name__ == "__main__":
    main()
