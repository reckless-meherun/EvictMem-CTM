"""Summarize completed test results and plot F1 by event gap."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = {
    "system_logs": {"gru": "GRU", "ctm": "CTM", "evictmem": "EvictMem-CTM"},
    "assoc_recall": {"gru": "GRU", "ctm": "CTM",
                     "ctm_capacity": "CTM-capacity", "evictmem": "EvictMem-CTM"},
}
GAPS = {"system_logs": (4, 8, 16, 32, 48),
        "assoc_recall": (16, 64, 128, 256, 384)}


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
    parser.add_argument("--dataset", choices=tuple(MODELS), default="system_logs")
    args = parser.parse_args()

    runs_dir = args.runs_dir / args.dataset if args.dataset == "assoc_recall" else args.runs_dir
    results_dir = args.results_dir / args.dataset if args.dataset == "assoc_recall" else args.results_dir
    gaps = GAPS[args.dataset]
    summary_rows = []
    gap_rows = []
    for model, label in MODELS[args.dataset].items():
        path = runs_dir / f"{model}_seed{args.seed}.json"
        if not path.is_file():
            parser.error(f"Required result file is missing: {path}")
        with path.open() as source:
            result = json.load(source)
        try:
            if result["model"] != model or result["seed"] != args.seed or result.get("dataset", "system_logs") != args.dataset:
                raise ValueError("dataset, model, or seed does not match filename")
            test = result["metrics"]["test"]
            summary_rows.append({"model": label, "accuracy": test["accuracy"],
                                 "f1": test["f1"],
                                 "parameter_count": result["parameter_count"],
                                 "training_time_seconds": result["training_time_seconds"]})
            gap_rows.append({"model": label, **{
                f"G={gap}": test["per_gap"][str(gap)]["f1"] for gap in gaps}})
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"Invalid result file {path}: {error}")

    results_dir.mkdir(parents=True, exist_ok=True)
    write_table(results_dir / "summary.csv",
                ["model", "accuracy", "f1", "parameter_count", "training_time_seconds"],
                summary_rows)
    write_table(results_dir / "gap_f1.csv",
                ["model", *(f"G={gap}" for gap in gaps)], gap_rows)

    fig, ax = plt.subplots()
    for row in gap_rows:
        ax.plot(gaps, [row[f"G={gap}"] for gap in gaps], marker="o", label=row["model"])
    ax.set(xlabel="Event gap", ylabel="Test F1", xticks=gaps, ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "f1_vs_gap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
