"""Run a sequence-classification experiment."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evictmem_ctm.data.torch_dataset import make_dataloaders
from evictmem_ctm.experiment import (count_parameters, evaluate, fit,
                                     load_checkpoint, seed_everything)
from evictmem_ctm.models.ctm import SequenceCTM
from evictmem_ctm.models.gru import GRUClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gru", "ctm"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64, help="CTM neuron count")
    parser.add_argument("--memory-length", type=int, default=5, help="CTM NLM FIFO length")
    parser.add_argument("--memory-hidden-dim", type=int, default=16)
    parser.add_argument("--n-synch-out", type=int, default=64)
    parser.add_argument("--pairing-seed", type=int, default=None,
                        help="CTM pair seed; defaults to --seed")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/system_logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.seed < 0 or args.lr <= 0 or args.patience < 1:
        parser.error("--seed must be nonnegative, --lr positive, and --patience positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    seed_everything(args.seed)
    loaders = make_dataloaders(args.data_dir, batch_size=args.batch_size, seed=args.seed)
    if args.model == "gru":
        model_config = {"vocab_size": 16, "embedding_dim": args.embedding_dim,
                        "hidden_size": 64, "num_classes": 2, "num_layers": 1,
                        "dropout": 0.0}
        model = GRUClassifier(**model_config)
    else:
        model_config = {"vocab_size": 16, "embedding_dim": args.embedding_dim,
                        "d_model": args.d_model, "memory_length": args.memory_length,
                        "memory_hidden_dim": args.memory_hidden_dim,
                        "n_synch_out": args.n_synch_out, "num_classes": 2,
                        "dropout": 0.0, "deep_nlm": True,
                        "pairing_seed": args.seed if args.pairing_seed is None else args.pairing_seed}
        model = SequenceCTM(**model_config)
    model = model.to(device)
    checkpoint_path = args.checkpoint_dir / f"{args.model}_seed{args.seed}.pt"
    summary = fit(model, loaders["train"], loaders["val"], device, checkpoint_path,
                  max_epochs=args.epochs, learning_rate=args.lr, weight_decay=0.0,
                  patience=args.patience, model_name=args.model,
                  model_config=model_config, seed=args.seed)
    load_checkpoint(checkpoint_path, model, device,
                    expected_model_name=args.model, expected_model_config=model_config)
    metrics = {split: evaluate(model, loaders[split], device)
               for split in ("train", "val", "test")}
    result = {
        "model": args.model,
        "seed": args.seed,
        "parameter_count": count_parameters(model),
        **summary,
        "metrics": metrics,
        "hyperparameters": {
            "model_config": model_config, "optimizer": "AdamW", "learning_rate": args.lr,
            "weight_decay": 0.0, "loss": "CrossEntropyLoss",
            "max_epochs": args.epochs, "early_stopping_patience": args.patience,
            "batch_size": args.batch_size, "num_workers": 0, "seed": args.seed,
            "device": device_name, "data_dir": str(args.data_dir),
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.model}_seed{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved results to {result_path} and best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
