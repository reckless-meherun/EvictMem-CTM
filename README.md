# EvictMem-CTM

## Synthetic system logs

Create and activate the Conda environment, then generate, inspect, and test the
20,000-example dataset:

```bash
conda env create -f environment.yml
conda activate evictmem-ctm

python scripts/generate_system_logs.py --seed 42
python scripts/inspect_system_logs.py
python -m unittest discover -s tests -v
```

Use `--output-dir` and `--data-dir` to choose another location.
The input `sequences` contain only token IDs; labels, gaps, pattern IDs, and
negative types are separate arrays. Positions are zero-based and the gap is
`position(B) - position(A)`. Positive examples have `negative_types=-1`;
benign negatives have `pattern_ids=-1`. Other negative pattern IDs identify
the A endpoint's pattern, except B-only examples, where they identify B's pattern.
Negative types 0–4 mean reversed, mismatched, A-only, B-only, and benign.
For one-endpoint and benign negatives, the assigned gap is a generation stratum;
it cannot be reconstructed from the visible event tokens alone.

## GRU baseline

Run the single-layer GRU experiment after activating the Conda environment:

```bash
python scripts/run_experiment.py --model gru
```

Defaults: 16-token vocabulary, 32-dimensional embeddings, 64 hidden units,
one GRU layer, no dropout, AdamW at `1e-3`, batch size 128, up to 20 epochs,
and early stopping after 3 epochs without validation-F1 improvement. The
best checkpoint goes to `checkpoints/` and metrics go to `runs/`.
