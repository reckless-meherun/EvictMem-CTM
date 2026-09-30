"""Small checks for model interfaces and shared experiment behavior."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evictmem_ctm.experiment import (binary_metrics, evaluate, fit, load_checkpoint,
                                     save_checkpoint)
from evictmem_ctm.models.ctm import SequenceCTM
from evictmem_ctm.models.ctm_components import (RandomPairSynchronisation,
                                                advance_pre_activation_trace)
from evictmem_ctm.models.gru import GRUClassifier
from run_experiment import closest_capacity_hidden_dim, resolve_training_policy


class EchoFirstToken(nn.Module):
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        prediction = sequence[:, 0].float()
        return torch.stack((1 - prediction, prediction), dim=1)


class NumberedPreActivations(nn.Module):
    """Return a known pre-activation at each recurrent tick."""

    def __init__(self) -> None:
        super().__init__()
        self.tick = 0

    def forward(self, synapse_input: torch.Tensor) -> torch.Tensor:
        self.tick += 1
        return synapse_input.new_full((synapse_input.shape[0], 2), self.tick)


class ModelTests(unittest.TestCase):
    def test_model_output_shapes_and_finite_ctm_logits(self) -> None:
        sequence = torch.randint(0, 16, (2, 64), dtype=torch.long)
        with torch.inference_mode():
            self.assertEqual(GRUClassifier()(sequence).shape, (2, 2))
            logits = SequenceCTM()(sequence)
            self.assertEqual(logits.shape, (2, 2))
            self.assertTrue(bool(torch.isfinite(logits).all()))
            self.assertEqual(SequenceCTM(memory_length=3)(sequence).shape, (2, 2))

    def test_evicted_memory_uses_only_real_evictions(self) -> None:
        sequence = torch.zeros(2, 3, dtype=torch.long)
        for enabled in (False, True):
            model = SequenceCTM(d_model=2, memory_length=2, n_synch_out=2,
                                evicted_memory=enabled)
            self.assertEqual(model.alpha, 0.95)
            model.synapse = NumberedPreActivations()
            with torch.no_grad():
                model.initial_state.pre_activation_trace.fill_(7)
            nlm_inputs = []
            hook = model.nlm.register_forward_pre_hook(
                lambda _module, inputs: nlm_inputs.append(inputs[0].detach().clone()))
            try:
                self.assertEqual(model(sequence).shape, (2, 2))
            finally:
                hook.remove()
            self.assertEqual(len(nlm_inputs), sequence.shape[1])
            self.assertEqual([value.shape[-1] for value in nlm_inputs],
                             [3 if enabled else 2] * 3)
            for tick, expected_trace in enumerate(((7, 1), (1, 2), (2, 3))):
                torch.testing.assert_close(
                    nlm_inputs[tick][:, :, :2],
                    torch.tensor(expected_trace, dtype=torch.float).expand(2, 2, 2))
            if enabled:
                torch.testing.assert_close(nlm_inputs[0][:, :, 2], torch.zeros(2, 2))
                torch.testing.assert_close(nlm_inputs[1][:, :, 2], torch.zeros(2, 2))
                torch.testing.assert_close(nlm_inputs[2][:, :, 2],
                                           torch.full((2, 2), 0.05))

    def test_fifo_keeps_only_latest_pre_activations(self) -> None:
        trace = torch.tensor([[[10., 11., 12.], [20., 21., 22.]]])
        updated = advance_pre_activation_trace(trace, torch.tensor([[13., 23.]]))
        torch.testing.assert_close(updated, torch.tensor([[[11., 12., 13.], [21., 22., 23.]]]))

    def test_pair_buffers_are_seeded_and_checkpointed(self) -> None:
        rng_before = torch.get_rng_state()
        RandomPairSynchronisation(64, 64, pairing_seed=7)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        first = SequenceCTM(pairing_seed=7).synchronisation
        second = SequenceCTM(pairing_seed=7).synchronisation
        self.assertIn("left_indices", dict(first.named_buffers()))
        self.assertIn("right_indices", dict(first.named_buffers()))
        for key in ("left_indices", "right_indices"):
            self.assertIn(key, first.state_dict())
            torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key])

    def test_decay_projection_preserves_gradient(self) -> None:
        synchronisation = RandomPairSynchronisation(1, 1)
        with torch.no_grad():
            synchronisation.decay_params.fill_(-1)
        output, _, _ = synchronisation(
            torch.ones(1, 1), torch.full((1, 1), 2.0), torch.ones(1, 1))
        self.assertEqual(synchronisation.decay_params.item(), 0.0)
        output.sum().backward()
        gradient = synchronisation.decay_params.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertNotEqual(gradient.item(), 0.0)

    def test_capacity_control_matches_nearest_vanilla_width(self) -> None:
        config = {"vocab_size": 74, "embedding_dim": 32, "d_model": 64,
                  "memory_length": 5, "memory_hidden_dim": 16,
                  "n_synch_out": 64, "num_classes": 2, "dropout": 0.0,
                  "deep_nlm": True, "pairing_seed": 42}
        hidden, target = closest_capacity_hidden_dim(config)
        self.assertGreater(hidden, config["memory_hidden_dim"])
        control = SequenceCTM(**{**config, "memory_hidden_dim": hidden})
        self.assertFalse(control.evicted_memory)
        self.assertEqual(target, sum(p.numel() for p in SequenceCTM(
            **config, evicted_memory=True).parameters()))
        distances = [abs(sum(p.numel() for p in SequenceCTM(
            **{**config, "memory_hidden_dim": size}).parameters()) - target)
            for size in range(12, 21)]
        self.assertEqual(abs(sum(p.numel() for p in control.parameters()) - target),
                         min(distances))

    def test_binary_metrics_and_per_gap_aggregation(self) -> None:
        actual = torch.tensor([0, 1, 0, 1])
        predicted = torch.tensor([0, 1, 1, 0])
        self.assertEqual(binary_metrics(predicted, actual), {"accuracy": 0.5, "f1": 0.5})

        labels = torch.tensor([0, 1] * 5)
        gaps = torch.tensor([gap for gap in (4, 8, 16, 32, 48) for _ in range(2)])
        predictions = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        sequences = torch.zeros(10, 64, dtype=torch.long)
        sequences[:, 0] = predictions
        loader = DataLoader(TensorDataset(sequences, labels, gaps), batch_size=3)
        metrics = evaluate(EchoFirstToken(), loader, torch.device("cpu"))
        self.assertEqual((metrics["accuracy"], metrics["f1"]), (0.8, 0.8))
        self.assertEqual(set(metrics["per_gap"]), {4, 8, 16, 32, 48})
        self.assertEqual(metrics["per_gap"][4]["f1"], 1.0)
        self.assertEqual(metrics["per_gap"][8]["f1"], 0.0)
        self.assertEqual(metrics["per_gap"][8]["accuracy"], 0.0)

    def test_dataset_training_defaults_and_overrides(self) -> None:
        self.assertEqual(resolve_training_policy("system_logs"),
                         {"max_epochs": 20, "min_epochs": 1,
                          "patience": 3, "selection_metric": "f1"})
        self.assertEqual(resolve_training_policy("assoc_recall"),
                         {"max_epochs": 50, "min_epochs": 20,
                          "patience": 10, "selection_metric": "accuracy"})
        self.assertEqual(resolve_training_policy("assoc_recall", epochs=7,
                         min_epochs=3, patience=2, selection_metric="f1"),
                         {"max_epochs": 7, "min_epochs": 3,
                          "patience": 2, "selection_metric": "f1"})
        for overrides in ({"epochs": 0}, {"min_epochs": 0}, {"patience": -1},
                          {"epochs": 5, "min_epochs": 6},
                          {"selection_metric": "loss"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_training_policy("assoc_recall", **overrides)

    def test_minimum_epochs_and_training_history(self) -> None:
        validations = [{"loss": 0.7, "accuracy": 0.6, "f1": 0.7}] + [
            {"loss": 0.8, "accuracy": 0.5, "f1": 0.6}] * 4
        model = nn.Linear(1, 2)
        with tempfile.TemporaryDirectory() as directory, \
                patch("evictmem_ctm.experiment.train_one_epoch", return_value=0.4), \
                patch("evictmem_ctm.experiment.evaluate", side_effect=validations):
            summary = fit(model, None, None, torch.device("cpu"),
                          Path(directory) / "best.pt", max_epochs=5, min_epochs=4,
                          patience=1, model_name="tiny", model_config={"size": 1},
                          selection_metric="accuracy")
        self.assertEqual(summary["epochs_completed"], 4)
        self.assertEqual(summary["best_epoch"], 1)
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["stop_reason"], "patience")
        self.assertEqual(len(summary["history"]), 4)
        self.assertEqual(set(summary["history"][0]),
                         {"epoch", "train_loss", "val_loss", "val_accuracy",
                          "val_f1", "selection_value"})
        self.assertEqual(summary["history"][0]["selection_value"], 0.6)

    def test_checkpoint_selection_uses_requested_metric(self) -> None:
        validations = [{"loss": 0.7, "accuracy": 0.5, "f1": 0.8},
                       {"loss": 0.6, "accuracy": 0.7, "f1": 0.6}]
        for metric, expected_epoch in (("accuracy", 2), ("f1", 1)):
            model = nn.Linear(1, 2)
            with tempfile.TemporaryDirectory() as directory, \
                    patch("evictmem_ctm.experiment.train_one_epoch", return_value=0.4), \
                    patch("evictmem_ctm.experiment.evaluate", side_effect=validations):
                path = Path(directory) / "best.pt"
                summary = fit(model, None, None, torch.device("cpu"), path,
                              max_epochs=2, min_epochs=2, patience=2,
                              model_name="tiny", model_config={"size": 1},
                              selection_metric=metric)
                checkpoint = load_checkpoint(path, nn.Linear(1, 2), torch.device("cpu"),
                                             expected_model_name="tiny",
                                             expected_model_config={"size": 1})
            self.assertEqual(summary["best_epoch"], expected_epoch)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertFalse(summary["stopped_early"])
            self.assertEqual(summary["stop_reason"], "max_epochs")
            self.assertEqual(checkpoint["epoch"], expected_epoch)
            self.assertEqual(checkpoint["selection_metric"], metric)
            self.assertEqual(checkpoint["selection_value"], summary["best_selection_value"])
            self.assertEqual(checkpoint["validation_accuracy"],
                             summary["best_validation_accuracy"])
            self.assertEqual(checkpoint["validation_f1"], summary["best_validation_f1"])

    def test_checkpoint_has_model_identity_and_reconstructs(self) -> None:
        config = {"vocab_size": 16, "embedding_dim": 32, "hidden_size": 64,
                  "num_classes": 2, "num_layers": 1, "dropout": 0.0}
        model = GRUClassifier(**config)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, 2, 0.75, "gru", config, 42)
            reconstructed = GRUClassifier(**config)
            checkpoint = load_checkpoint(path, reconstructed, torch.device("cpu"),
                                         expected_model_name="gru", expected_model_config=config)
            self.assertEqual((checkpoint["model_name"], checkpoint["seed"],
                              checkpoint["epoch"], checkpoint["validation_f1"]),
                             ("gru", 42, 2, 0.75))
            self.assertIn("optimizer_state", checkpoint)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reconstructed.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
