"""Summarize completed test results and plot F1 by event gap."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = {"gru": "GRU", "ctm": "CTM", "evictmem": "EvictMem-CTM"}
GAPS = (4, 8, 16, 32, 48)


def write_table(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary_rows = []
    gap_rows = []
    for model, label in MODELS.items():
        path = args.runs_dir / f"{model}_seed{args.seed}.json"
        if not path.is_file():
            parser.error(f"Required result file is missing: {path}")
        with path.open() as source:
            result = json.load(source)
        try:
            if result["model"] != model or result["seed"] != args.seed:
                raise ValueError("model or seed does not match filename")
            test = result["metrics"]["test"]
            summary_rows.append({"model": label, "accuracy": test["accuracy"],
                                 "f1": test["f1"],
                                 "parameter_count": result["parameter_count"],
                                 "training_time_seconds": result["training_time_seconds"]})
            gap_rows.append({"model": label, **{
                f"G={gap}": test["per_gap"][str(gap)]["f1"] for gap in GAPS}})
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"Invalid result file {path}: {error}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.results_dir / "summary.csv",
                ["model", "accuracy", "f1", "parameter_count", "training_time_seconds"],
                summary_rows)
    write_table(args.results_dir / "gap_f1.csv",
                ["model", *(f"G={gap}" for gap in GAPS)], gap_rows)

    fig, ax = plt.subplots()
    for row in gap_rows:
        ax.plot(GAPS, [row[f"G={gap}"] for gap in GAPS], marker="o", label=row["model"])
    ax.set(xlabel="Event gap", ylabel="Test F1", xticks=GAPS, ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results_dir / "f1_vs_gap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
