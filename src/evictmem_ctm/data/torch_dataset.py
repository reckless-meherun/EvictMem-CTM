"""PyTorch access to token-sequence NPZ splits."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evictmem_ctm.data import assoc_recall, system_logs


DATASETS = {
    "system_logs": {"length": system_logs.LENGTH, "vocab_size": len(system_logs.TOKENS),
                    "gaps": system_logs.GAPS, "split_sizes": system_logs.SPLIT_SIZES},
    "assoc_recall": {"length": assoc_recall.LENGTH, "vocab_size": assoc_recall.VOCAB_SIZE,
                     "gaps": assoc_recall.GAPS, "split_sizes": assoc_recall.SPLIT_SIZES},
}


class SystemLogDataset(Dataset):
    """Return token IDs, label, and evaluation-only gap metadata."""

    def __init__(self, path: Path, dataset: str = "system_logs") -> None:
        config = DATASETS[dataset]
        with np.load(path, allow_pickle=False) as archive:
            self.sequences = torch.from_numpy(archive["sequences"].astype(np.int64))
            self.labels = torch.from_numpy(archive["labels"].astype(np.int64))
            self.gaps = torch.from_numpy(archive["gaps"].astype(np.int64))
        if self.sequences.ndim != 2 or self.sequences.shape[1] != config["length"]:
            raise ValueError(f"{path}: expected sequences with length {config['length']}")
        if self.labels.shape != (len(self.sequences),) or self.gaps.shape != (len(self.sequences),):
            raise ValueError(f"{path}: labels and gaps must have shape [N]")
        if not bool(torch.all((self.sequences >= 0) & (self.sequences < config["vocab_size"]))):
            raise ValueError(f"{path}: token IDs outside vocabulary")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index], self.gaps[index]


def make_dataloaders(data_dir: Path, batch_size: int = 128, num_workers: int = 0,
                     seed: int = 42, dataset: str = "system_logs") -> dict[str, DataLoader]:
    """Shuffle training examples reproducibly; preserve evaluation order."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(SystemLogDataset(data_dir / f"{split}.npz", dataset),
                          batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, generator=generator if split == "train" else None)
        for split in ("train", "val", "test")
    }
