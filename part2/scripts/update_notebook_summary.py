from __future__ import annotations

import json
import uuid

from part2_utils import PYTHON_EXE, ROOT


def _cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    nb_path = ROOT / "part2.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    phase_summary_path = ROOT / "outputs" / "phase_summary.md"
    reproduce_path = ROOT / "outputs" / "REPRODUCE.md"
    snippet_path = ROOT / "outputs" / "final_report_snippet.md"
    phase_summary = phase_summary_path.read_text(encoding="utf-8") if phase_summary_path.exists() else "Phase summary not generated yet."
    reproduce = reproduce_path.read_text(encoding="utf-8") if reproduce_path.exists() else "REPRODUCE.md not generated yet."
    snippet = snippet_path.read_text(encoding="utf-8") if snippet_path.exists() else ""

    compact_lines = []
    keep = False
    for line in phase_summary.splitlines():
        if line.startswith("## Transformer OOF Results") or line.startswith("## Stacker") or line.startswith("## Duplicate Transfer") or line.startswith("## Final Selection"):
            keep = True
            compact_lines.append(line)
            continue
        if line.startswith("## ") and keep:
            keep = False
        if keep and len(compact_lines) < 80:
            compact_lines.append(line)

    markdown = "\n".join(
        [
            "# Final High-Score Pipeline",
            "",
            "Heavy training and feature generation are implemented in `scripts/` so the notebook remains readable and reproducible.",
            "",
            "## Method Summary",
            snippet.strip(),
            "",
            "## Result Summary",
            "\n".join(compact_lines).strip(),
            "",
            "## Reproduction",
            f"All commands use `{PYTHON_EXE}`. See `outputs/REPRODUCE.md` for the full command sequence.",
            "",
            "```powershell",
            r"cd C:\Coding\Intro_to_AI\project1-part2\part2",
            f'& "{PYTHON_EXE}" scripts\\preflight.py',
            f'& "{PYTHON_EXE}" scripts\\run_oof.py --model-name distilbert-base-uncased --model-tag distilbert --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune --augment-heuristic-splitting',
            f'& "{PYTHON_EXE}" scripts\\run_oof.py --model-name microsoft/deberta-v3-base --model-tag deberta_v3 --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune',
            f'& "{PYTHON_EXE}" scripts\\run_oof.py --model-name roberta-base --model-tag roberta --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune',
            f'& "{PYTHON_EXE}" scripts\\classical_oof.py',
            f'& "{PYTHON_EXE}" scripts\\endpoint_features.py',
            f'& "{PYTHON_EXE}" scripts\\retrieval_features.py',
            f'& "{PYTHON_EXE}" scripts\\stack_oof.py',
            f'& "{PYTHON_EXE}" scripts\\duplicate_transfer.py',
            f'& "{PYTHON_EXE}" scripts\\summarize.py',
            "```",
            "",
            "Final selected CSV: `submissions/submission_final.csv`.",
            "",
            "This section documents the final pipeline only; it does not embed train/test dumps, hidden labels, LLM labels, or public leaderboard tuning.",
        ]
    )

    cells = nb.setdefault("cells", [])
    for cell in cells:
        if cell.get("cell_type") == "markdown" and "Final High-Score Pipeline" in "".join(cell.get("source", [])):
            cell["source"] = markdown.splitlines(keepends=True)
            break
    else:
        cells.append(_cell(markdown))
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(nb_path.read_text(encoding="utf-8"))
    print(f"updated {nb_path}")


if __name__ == "__main__":
    main()
