"""Deterministic paired associative recall with long-range interference."""

import argparse
import json
from pathlib import Path

import numpy as np


N_KEYS = 32
N_ASSIGNMENTS = 24
LENGTH = 512
GAPS = (16, 64, 128, 256, 384)
SPLIT_SIZES = {"train": 14000, "val": 3000, "test": 3000}
VALUE_0, VALUE_1 = 64, 65
NOISE_START = 66
VOCAB_SIZE = 74
FIELDS = ("sequences", "labels", "gaps", "query_key_ids", "pair_ids")
TOKENS = ([f"K{i}" for i in range(N_KEYS)]
          + [f"Q{i}" for i in range(N_KEYS)]
          + ["VALUE_0", "VALUE_1"]
          + [f"NOISE_{i}" for i in range(8)])


def _assignment_positions(rng: np.random.Generator, target_value_pos: int) -> np.ndarray:
    occupied = np.zeros(LENGTH, dtype=bool)
    occupied[target_value_pos - 1:target_value_pos + 1] = True
    occupied[-1] = True
    positions = [target_value_pos]
    for value_pos in rng.permutation(np.arange(1, LENGTH - 1)):
        value_pos = int(value_pos)
        if occupied[value_pos - 1] or occupied[value_pos]:
            continue
        occupied[value_pos - 1:value_pos + 1] = True
        positions.append(value_pos)
        if len(positions) == N_ASSIGNMENTS:
            break
    if len(positions) != N_ASSIGNMENTS:
        raise RuntimeError("Could not place distinct assignments")
    return np.asarray(positions, dtype=np.int16)


def _generate_split(size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if size % (2 * len(GAPS)):
        raise ValueError("Split size must be divisible by twice the number of gaps")
    sequences, labels, gaps, query_keys, pair_ids = [], [], [], [], []
    pair_id = 0
    for gap in GAPS:
        pairs_per_gap = size // (2 * len(GAPS))
        target_keys = np.tile(np.arange(N_KEYS), (pairs_per_gap + N_KEYS - 1) // N_KEYS)[:pairs_per_gap]
        rng.shuffle(target_keys)
        for target_key in target_keys:
            target_key = int(target_key)
            keys = np.concatenate(([target_key], rng.choice(
                np.delete(np.arange(N_KEYS), target_key), N_ASSIGNMENTS - 1, replace=False)))
            decoy_index = int(rng.integers(1, N_ASSIGNMENTS))
            value_positions = _assignment_positions(rng, LENGTH - 1 - gap)
            values = np.zeros(N_ASSIGNMENTS, dtype=np.uint8)
            values[decoy_index] = 1
            other_indices = np.asarray([i for i in range(1, N_ASSIGNMENTS) if i != decoy_index])
            values[rng.choice(other_indices, 11, replace=False)] = 1
            base = rng.integers(NOISE_START, VOCAB_SIZE, size=LENGTH, dtype=np.uint8)
            base[value_positions - 1] = keys
            base[value_positions] = VALUE_0 + values
            base[-1] = N_KEYS + target_key
            for label in (0, 1):
                sequence = base.copy()
                sequence[value_positions[0]] = VALUE_0 + label
                sequence[value_positions[decoy_index]] = VALUE_1 - label
                sequences.append(sequence)
                labels.append(label)
                gaps.append(gap)
                query_keys.append(target_key)
                pair_ids.append(pair_id)
            pair_id += 1
    order = rng.permutation(size)
    return {
        "sequences": np.stack(sequences)[order],
        "labels": np.asarray(labels, dtype=np.uint8)[order],
        "gaps": np.asarray(gaps, dtype=np.uint16)[order],
        "query_key_ids": np.asarray(query_keys, dtype=np.uint8)[order],
        "pair_ids": np.asarray(pair_ids, dtype=np.int32)[order],
    }


def generate(seed: int = 42) -> dict[str, dict[str, np.ndarray]]:
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    streams = np.random.SeedSequence(seed).spawn(len(SPLIT_SIZES))
    return {name: _generate_split(size, np.random.default_rng(stream))
            for (name, size), stream in zip(SPLIT_SIZES.items(), streams)}


def save_dataset(directory: Path, splits: dict[str, dict[str, np.ndarray]], seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, arrays in splits.items():
        np.savez_compressed(directory / f"{name}.npz", **arrays)
    (directory / "metadata.json").write_text(json.dumps({
        "dataset": "assoc_recall", "seed": seed, "sequence_length": LENGTH,
        "vocab_size": VOCAB_SIZE, "gaps": GAPS, "split_sizes": SPLIT_SIZES,
        "vocabulary": {token: i for i, token in enumerate(TOKENS)},
    }, indent=2) + "\n")


def load_dataset(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for name in SPLIT_SIZES:
        with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(FIELDS):
                raise ValueError(f"{name}: unexpected NPZ fields")
            result[name] = {field: archive[field] for field in FIELDS}
    return result


def validate(splits: dict[str, dict[str, np.ndarray]], directory: Path | None = None,
             check_regeneration: bool = True) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(set(splits) == set(SPLIT_SIZES), "Wrong split names")
    seen = set()
    for name, arrays in splits.items():
        size = SPLIT_SIZES[name]
        require(set(arrays) == set(FIELDS), f"{name}: wrong fields")
        for field in FIELDS:
            require(arrays[field].shape == ((size, LENGTH) if field == "sequences" else (size,)),
                    f"{name}.{field}: wrong shape")
        sequences, labels, gaps = (arrays[field] for field in ("sequences", "labels", "gaps"))
        query_keys, pair_ids = arrays["query_key_ids"], arrays["pair_ids"]
        require(np.issubdtype(sequences.dtype, np.integer) and
                bool(np.all(sequences < VOCAB_SIZE)) and bool(np.all(sequences >= 0)),
                f"{name}: invalid token IDs")
        require(bool(np.all(np.isin(labels, (0, 1)))) and int(labels.sum()) == size // 2,
                f"{name}: class imbalance")
        require(bool(np.all(np.isin(gaps, GAPS))), f"{name}: invalid gap")
        require(bool(np.all((query_keys >= 0) & (query_keys < N_KEYS))), f"{name}: invalid query key")
        require(np.array_equal(sequences[:, -1], N_KEYS + query_keys), f"{name}: wrong final query")
        for gap in GAPS:
            selected = gaps == gap
            require(int(selected.sum()) == size // len(GAPS), f"{name}: gap imbalance")
            for label in (0, 1):
                require(int(np.sum(selected & (labels == label))) == size // (2 * len(GAPS)),
                        f"{name}: gap/label imbalance")
            counts = np.bincount(query_keys[selected], minlength=N_KEYS)
            require(int(counts.max() - counts.min()) <= 2, f"{name}: query-key imbalance")
        pair_rows = {}
        for index, sequence in enumerate(sequences):
            key_positions = np.flatnonzero(sequence < N_KEYS)
            value_positions = np.flatnonzero((sequence == VALUE_0) | (sequence == VALUE_1))
            target_key = int(query_keys[index])
            target_positions = np.flatnonzero(sequence == target_key)
            require(len(key_positions) == N_ASSIGNMENTS and
                    len(np.unique(sequence[key_positions])) == N_ASSIGNMENTS and
                    len(value_positions) == N_ASSIGNMENTS and
                    np.array_equal(value_positions, key_positions + 1),
                    f"{name}[{index}]: invalid assignments")
            require(not bool(np.any((sequence[:-1] >= N_KEYS) & (sequence[:-1] < VALUE_0))) and
                    bool(np.all(sequence[np.setdiff1d(np.arange(LENGTH),
                         np.concatenate((key_positions, value_positions, [LENGTH - 1])))] >= NOISE_START)),
                    f"{name}[{index}]: unexpected tokens")
            require(len(target_positions) == 1, f"{name}[{index}]: target key not unique")
            target_value_pos = int(target_positions[0]) + 1
            require(LENGTH - 1 - target_value_pos == int(gaps[index]),
                    f"{name}[{index}]: wrong target gap")
            require(int(sequence[target_value_pos]) == VALUE_0 + int(labels[index]),
                    f"{name}[{index}]: wrong target label")
            key = sequence.tobytes()
            require(key not in seen, f"{name}[{index}]: duplicate sequence")
            seen.add(key)
            pair_rows.setdefault(int(pair_ids[index]), []).append(index)
        require(len(pair_rows) == size // 2, f"{name}: wrong pair count")
        for pair_id, indices in pair_rows.items():
            require(len(indices) == 2, f"{name}: pair {pair_id} has wrong size")
            first, second = indices
            a, b = sequences[first], sequences[second]
            differing = np.flatnonzero(a != b)
            target_pos = LENGTH - 1 - int(gaps[first])
            require(set(labels[indices]) == {0, 1} and gaps[first] == gaps[second] and
                    query_keys[first] == query_keys[second] and len(differing) == 2 and
                    target_pos in differing and
                    set(map(int, a[differing])) == {VALUE_0, VALUE_1} and
                    set(map(int, b[differing])) == {VALUE_0, VALUE_1} and
                    np.array_equal(np.bincount(a, minlength=VOCAB_SIZE),
                                   np.bincount(b, minlength=VOCAB_SIZE)),
                    f"{name}: pair {pair_id} is not a matched target/decoy swap")
    if directory is not None:
        recorded = json.loads((directory / "metadata.json").read_text())
        require(recorded == {"dataset": "assoc_recall", "seed": recorded["seed"],
                 "sequence_length": LENGTH, "vocab_size": VOCAB_SIZE,
                 "gaps": list(GAPS), "split_sizes": SPLIT_SIZES,
                 "vocabulary": {token: i for i, token in enumerate(TOKENS)}},
                "Metadata differs from specification")
        if check_regeneration:
            regenerated = generate(recorded["seed"])
            for name in SPLIT_SIZES:
                for field in FIELDS:
                    require(np.array_equal(splits[name][field], regenerated[name][field]),
                            f"{name}.{field}: deterministic regeneration differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/assoc_recall"))
    args = parser.parse_args()
    generated = generate(args.seed)
    validate(generated, check_regeneration=False)
    save_dataset(args.output_dir, generated, args.seed)
    validate(generated, args.output_dir)
    print(f"Generated associative recall dataset in {args.output_dir}")
