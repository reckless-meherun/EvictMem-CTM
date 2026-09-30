"""Structural, pairing, and reproducibility checks for associative recall."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evictmem_ctm.data import assoc_recall as task
from evictmem_ctm.data.torch_dataset import SystemLogDataset


class AssociativeRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = task.generate(42)

    def test_full_dataset_and_saved_metadata(self) -> None:
        task.validate(self.splits, check_regeneration=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            task.save_dataset(path, self.splits, 42)
            task.validate(task.load_dataset(path), path)

    def test_balanced_matched_pairs_and_target_gap(self) -> None:
        for name, arrays in self.splits.items():
            size = task.SPLIT_SIZES[name]
            self.assertEqual(arrays["sequences"].shape, (size, task.LENGTH))
            self.assertEqual(int(arrays["labels"].sum()), size // 2)
            for gap in task.GAPS:
                selected = arrays["gaps"] == gap
                self.assertEqual(int(selected.sum()), size // len(task.GAPS))
            for indices in np.argsort(arrays["pair_ids"]).reshape(-1, 2):
                self.assertEqual(arrays["pair_ids"][indices[0]], arrays["pair_ids"][indices[1]])
                self.assertEqual(set(arrays["labels"][indices]), {0, 1})
                a, b = arrays["sequences"][indices]
                self.assertEqual(int(np.sum(a != b)), 2)
                np.testing.assert_array_equal(np.bincount(a, minlength=task.VOCAB_SIZE),
                                              np.bincount(b, minlength=task.VOCAB_SIZE))
                for index in indices:
                    sequence = arrays["sequences"][index]
                    target = int(arrays["query_key_ids"][index])
                    self.assertEqual(int(sequence[-1]), task.N_KEYS + target)
                    key_positions = np.flatnonzero(sequence == target)
                    self.assertEqual(len(key_positions), 1)
                    value_pos = int(key_positions[0]) + 1
                    self.assertEqual(task.LENGTH - 1 - value_pos, int(arrays["gaps"][index]))
                    self.assertEqual(int(sequence[value_pos]), task.VALUE_0 + int(arrays["labels"][index]))

    def test_seed_reproducibility_and_independence(self) -> None:
        regenerated = task.generate(42)
        for split in self.splits:
            for field in task.FIELDS:
                np.testing.assert_array_equal(self.splits[split][field], regenerated[split][field])
        different = task.generate(43)
        self.assertFalse(np.array_equal(self.splits["train"]["sequences"],
                                        different["train"]["sequences"]))

    def test_loader_exposes_only_tokens_label_and_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(path, **{field: values[:2]
                                         for field, values in self.splits["train"].items()})
            dataset = SystemLogDataset(path, "assoc_recall")
            tokens, label, gap = dataset[0]
            self.assertEqual(tuple(tokens.shape), (task.LENGTH,))
            self.assertEqual(label.ndim, 0)
            self.assertEqual(gap.ndim, 0)

    def test_validator_rejects_unmatched_swap(self) -> None:
        corrupted = {name: arrays.copy() for name, arrays in self.splits.items()}
        corrupted["train"] = {field: values.copy() for field, values in self.splits["train"].items()}
        pair_id = int(corrupted["train"]["pair_ids"][0])
        indices = np.flatnonzero(corrupted["train"]["pair_ids"] == pair_id)
        negative = int(indices[np.flatnonzero(corrupted["train"]["labels"][indices] == 0)[0]])
        sequence = corrupted["train"]["sequences"][negative]
        value_positions = np.flatnonzero((sequence == task.VALUE_0) | (sequence == task.VALUE_1))
        decoy_pos = next(int(pos) for pos in value_positions
                         if int(pos) != task.LENGTH - 1 - int(corrupted["train"]["gaps"][negative]))
        sequence[decoy_pos] = task.VALUE_0 if sequence[decoy_pos] == task.VALUE_1 else task.VALUE_1
        with self.assertRaises(ValueError):
            task.validate(corrupted, check_regeneration=False)


if __name__ == "__main__":
    unittest.main()
