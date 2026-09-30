"""Tests for the saved dataset and its semantic validator."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evictmem_ctm.data.system_logs import generate, load_dataset, validate


class SystemLogTests(unittest.TestCase):
    def test_saved_dataset_and_regeneration(self) -> None:
        validate(load_dataset(ROOT / "data/system_logs"), ROOT / "data/system_logs")

    def test_seed_reproducibility_and_independence(self) -> None:
        first, second = generate(42), generate(42)
        for split in first:
            for field in first[split]:
                np.testing.assert_array_equal(first[split][field], second[split][field])
        self.assertFalse(np.array_equal(first["train"]["sequences"], generate(43)["train"]["sequences"]))

    def test_validator_rejects_hidden_malicious_pair(self) -> None:
        data = generate(42)
        index = int(np.flatnonzero(data["train"]["negative_types"] == 4)[0])
        data["train"]["sequences"][index, 0:2] = [0, 1]
        with self.assertRaisesRegex(ValueError, "negative contains malicious pair"):
            validate(data, check_regeneration=False)


if __name__ == "__main__":
    unittest.main()
