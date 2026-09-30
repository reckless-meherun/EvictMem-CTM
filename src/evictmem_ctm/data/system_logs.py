"""Deterministic synthetic system logs with controlled long-range pairs."""

import json
from pathlib import Path

import numpy as np

TOKENS = (
    "SCAN", "ADMIN_ACCESS", "FAILED_LOGIN", "SUDO", "DOWNLOAD", "EXECUTE",
    "CONFIG_CHANGE", "SERVICE_RESTART", "LOGIN_OK", "READ", "WRITE",
    "HEARTBEAT", "DNS_QUERY", "FILE_OPEN", "PROCESS_LIST", "LOGOUT",
)
PATTERNS = ((0, 1), (2, 3), (4, 5), (6, 7))
NEGATIVE_TYPES = ("reversed", "mismatched", "A_only", "B_only", "benign")
SPLIT_SIZES = {"train": 14000, "val": 3000, "test": 3000}
GAPS = (4, 8, 16, 32, 48)
LENGTH = 64
FIELDS = ("sequences", "labels", "gaps", "pattern_ids", "negative_types")
DTYPES = {"sequences": np.dtype("uint8"), "labels": np.dtype("uint8"),
          "gaps": np.dtype("uint8"), "pattern_ids": np.dtype("int8"),
          "negative_types": np.dtype("int8")}


def metadata(seed: int, split_sizes: dict[str, int] = SPLIT_SIZES) -> dict:
    return {
        "sequence_length": LENGTH, "allowed_gaps": list(GAPS),
        "malicious_patterns": [[TOKENS[a], TOKENS[b]] for a, b in PATTERNS],
        "split_sizes": split_sizes, "random_seed": seed,
        "vocabulary": {token: index for index, token in enumerate(TOKENS)},
        "gap_definition": "G = position(B) - position(A); positions are zero-based",
        "negative_generation_rules": {
            "background": "Only the eight benign tokens (IDs 8-15)",
            "categories": list(NEGATIVE_TYPES),
            "reversed": "B at i, matching A at i+G",
            "mismatched": "A at i, B from a different pattern at i+G",
            "A_only": "A at i; benign at i+G",
            "B_only": "benign at i; B at i+G",
            "benign": "All tokens benign",
            "pattern_id": "Positive: matched pattern; negative: A pattern, or B pattern for B_only; -1 for benign",
            "negative_type": "-1 for positives; 0-4 index into categories for negatives",
        },
    }


def _generate_split(size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if size % (2 * len(GAPS) * len(NEGATIVE_TYPES)):
        raise ValueError("Each split size must be divisible by 50 for exact gap, class, and negative-type balance")
    rows = []
    for gap in GAPS:
        per_class = size // (2 * len(GAPS))
        for label, category, count in [(1, -1, per_class)] + [
            (0, category, per_class // len(NEGATIVE_TYPES))
            for category in range(len(NEGATIVE_TYPES))
        ]:
            pattern_ids = np.tile(np.arange(len(PATTERNS), dtype=np.int8),
                                  (count + len(PATTERNS) - 1) // len(PATTERNS))[:count]
            rng.shuffle(pattern_ids)
            for pattern_id in pattern_ids:
                start = int(rng.integers(0, LENGTH - gap))
                sequence = rng.integers(8, len(TOKENS), size=LENGTH, dtype=np.uint8)
                first, second = PATTERNS[int(pattern_id)]
                if label:
                    sequence[start], sequence[start + gap] = first, second
                elif category == 0:
                    sequence[start], sequence[start + gap] = second, first
                elif category == 1:
                    other = (int(pattern_id) + int(rng.integers(1, len(PATTERNS)))) % len(PATTERNS)
                    sequence[start], sequence[start + gap] = first, PATTERNS[other][1]
                elif category == 2:
                    sequence[start] = first
                elif category == 3:
                    sequence[start + gap] = second
                else:
                    pattern_id = -1
                rows.append((sequence, label, gap, pattern_id, category))
    order = rng.permutation(len(rows))
    return {
        "sequences": np.stack([rows[i][0] for i in order]),
        "labels": np.array([rows[i][1] for i in order], dtype=np.uint8),
        "gaps": np.array([rows[i][2] for i in order], dtype=np.uint8),
        "pattern_ids": np.array([rows[i][3] for i in order], dtype=np.int8),
        "negative_types": np.array([rows[i][4] for i in order], dtype=np.int8),
    }


def generate(seed: int = 42, split_sizes: dict[str, int] = SPLIT_SIZES) -> dict[str, dict[str, np.ndarray]]:
    """Use independent child RNG streams so splits do not share RNG state."""
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    streams = np.random.SeedSequence(seed).spawn(len(split_sizes))
    return {name: _generate_split(size, np.random.default_rng(stream))
            for (name, size), stream in zip(split_sizes.items(), streams)}


def save_dataset(directory: Path, splits: dict[str, dict[str, np.ndarray]], seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, arrays in splits.items():
        np.savez_compressed(directory / f"{name}.npz", **arrays)
    vocabulary = {"token_to_id": {token: i for i, token in enumerate(TOKENS)},
                  "id_to_token": {str(i): token for i, token in enumerate(TOKENS)}}
    (directory / "vocab.json").write_text(json.dumps(vocabulary, indent=2) + "\n")
    (directory / "metadata.json").write_text(
        json.dumps(metadata(seed, {name: len(data["labels"]) for name, data in splits.items()}), indent=2) + "\n")


def load_dataset(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for name in SPLIT_SIZES:
        with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(FIELDS):
                raise ValueError(f"{name}: unexpected NPZ fields: {archive.files}")
            result[name] = {field: archive[field] for field in FIELDS}
    return result


def validate(splits: dict[str, dict[str, np.ndarray]], directory: Path | None = None,
             check_regeneration: bool = True) -> None:
    """Raise ValueError on any structural, semantic, or reproducibility failure."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(set(splits) == set(SPLIT_SIZES), "Split names differ from specification")
    seen = set()
    for name, arrays in splits.items():
        size = SPLIT_SIZES[name]
        for field in FIELDS:
            expected_shape = (size, LENGTH) if field == "sequences" else (size,)
            require(arrays[field].shape == expected_shape, f"{name}.{field}: wrong shape")
            require(arrays[field].dtype == DTYPES[field], f"{name}.{field}: wrong dtype")
        sequences, labels, gaps = (arrays[key] for key in ("sequences", "labels", "gaps"))
        patterns, types = arrays["pattern_ids"], arrays["negative_types"]
        require(bool(np.all(sequences < len(TOKENS))), f"{name}: invalid token ID")
        require(bool(np.all(np.isin(labels, (0, 1)))), f"{name}: invalid label")
        require(bool(np.all(np.isin(gaps, GAPS))), f"{name}: invalid gap")
        require(int(labels.sum()) == size // 2, f"{name}: class imbalance")
        for gap in GAPS:
            selected = gaps == gap
            require(int(selected.sum()) == size // len(GAPS), f"{name}: gap {gap} imbalance")
            positives = selected & (labels == 1)
            negatives = selected & (labels == 0)
            require(int(positives.sum()) == size // (2 * len(GAPS)), f"{name}: positive gap {gap} imbalance")
            for category in range(len(NEGATIVE_TYPES)):
                require(int(np.sum(negatives & (types == category))) == size // (2 * len(GAPS) * len(NEGATIVE_TYPES)),
                        f"{name}: gap {gap}, negative type {category} imbalance")
            counts = [int(np.sum(positives & (patterns == p))) for p in range(len(PATTERNS))]
            require(max(counts) - min(counts) <= 1, f"{name}: gap {gap} pattern imbalance")
        require(bool(np.all(types[labels == 1] == -1)), f"{name}: positive negative_type must be -1")
        require(bool(np.all((types[labels == 0] >= 0) & (types[labels == 0] < 5))), f"{name}: invalid negative_type")
        for index, sequence in enumerate(sequences):
            label, gap, pattern_id, category = map(int, (labels[index], gaps[index], patterns[index], types[index]))
            positions = [np.flatnonzero(sequence == token) for token in range(8)]
            valid = [(p, int(a), int(b)) for p, (first, second) in enumerate(PATTERNS)
                     for a in positions[first] for b in positions[second] if a < b]
            endpoints = int(np.sum(sequence < 8))
            if label:
                require(endpoints == 2 and len(valid) == 1 and valid[0][0] == pattern_id and
                        valid[0][2] - valid[0][1] == gap, f"{name}[{index}]: invalid positive")
            else:
                require(not valid, f"{name}[{index}]: negative contains malicious pair")
                expected = (2, 2, 1, 1, 0)[category]
                require(endpoints == expected, f"{name}[{index}]: wrong endpoint count")
                if category == 4:
                    require(pattern_id == -1, f"{name}[{index}]: benign pattern_id must be -1")
                else:
                    require(0 <= pattern_id < 4, f"{name}[{index}]: invalid pattern_id")
                    a, b = PATTERNS[pattern_id]
                    if category == 0:
                        require(len(positions[a]) == len(positions[b]) == 1 and
                                int(positions[a][0] - positions[b][0]) == gap,
                                f"{name}[{index}]: invalid reversed pair")
                    elif category == 1:
                        require(len(positions[a]) == 1 and any(
                            len(positions[other_b]) == 1 and int(positions[other_b][0] - positions[a][0]) == gap
                            for p, (_, other_b) in enumerate(PATTERNS) if p != pattern_id),
                            f"{name}[{index}]: invalid mismatched pair")
                    elif category == 2:
                        require(len(positions[a]) == 1 and int(positions[a][0]) + gap < LENGTH,
                                f"{name}[{index}]: invalid A_only")
                    elif category == 3:
                        require(len(positions[b]) == 1 and int(positions[b][0]) >= gap,
                                f"{name}[{index}]: invalid B_only")
            key = sequence.tobytes()
            require(key not in seen, f"{name}[{index}]: duplicate sequence across or within splits")
            seen.add(key)
    if directory is not None:
        recorded = json.loads((directory / "metadata.json").read_text())
        seed = recorded.get("random_seed")
        require(recorded == metadata(seed), "metadata.json differs from dataset specification")
        vocab = json.loads((directory / "vocab.json").read_text())
        require(vocab["token_to_id"] == recorded["vocabulary"] and
                vocab["id_to_token"] == {str(i): token for i, token in enumerate(TOKENS)},
                "vocab.json disagrees with metadata")
        if check_regeneration:
            regenerated = generate(seed)
            for name in SPLIT_SIZES:
                for field in FIELDS:
                    require(np.array_equal(splits[name][field], regenerated[name][field]),
                            f"{name}.{field}: deterministic regeneration differs")
