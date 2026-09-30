"""Validate and inspect saved synthetic system logs."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evictmem_ctm.data.system_logs import (GAPS, NEGATIVE_TYPES, PATTERNS, SPLIT_SIZES,
                                            TOKENS, load_dataset, validate)


def describe(sequence: np.ndarray) -> str:
    return " ".join(f"{i}:{TOKENS[int(token)]}" for i, token in enumerate(sequence))


def show_example(split: dict[str, np.ndarray], index: int) -> None:
    sequence = split["sequences"][index]
    pattern_id = int(split["pattern_ids"][index])
    category = int(split["negative_types"][index])
    kind = "positive" if category == -1 else NEGATIVE_TYPES[category]
    endpoints = [(i, TOKENS[int(token)]) for i, token in enumerate(sequence) if token < 8]
    print(f"  {kind}, gap={int(split['gaps'][index])}, pattern={pattern_id}, endpoints={endpoints}")
    print(f"  {describe(sequence)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/system_logs"))
    args = parser.parse_args()
    splits = load_dataset(args.data_dir)
    validate(splits, args.data_dir)
    print("All validation checks passed.")
    for name in SPLIT_SIZES:
        data = splits[name]
        labels, gaps, patterns, types = (data[key] for key in ("labels", "gaps", "pattern_ids", "negative_types"))
        print(f"{name}: size={len(labels)}, positive={int(labels.sum())}, negative={int((labels == 0).sum())}")
        print("  gaps:", {gap: int((gaps == gap).sum()) for gap in GAPS})
        print("  positive patterns:", {f"{TOKENS[a]}->{TOKENS[b]}": int(((labels == 1) & (patterns == i)).sum())
                                        for i, (a, b) in enumerate(PATTERNS)})
        print("  negative types:", {kind: int((types == i).sum()) for i, kind in enumerate(NEGATIVE_TYPES)})
    train = splits["train"]
    print("Representative examples (train):")
    for gap in GAPS:
        show_example(train, int(np.flatnonzero((train["labels"] == 1) & (train["gaps"] == gap))[0]))
    for category in range(len(NEGATIVE_TYPES)):
        show_example(train, int(np.flatnonzero(train["negative_types"] == category)[0]))


if __name__ == "__main__":
    main()
