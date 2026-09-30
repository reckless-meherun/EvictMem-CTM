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

## Sequence classification experiments

After activating the Conda environment, the three experiment commands are:

```bash
python scripts/run_experiment.py --model gru
python scripts/run_experiment.py --model ctm
python scripts/run_experiment.py --model evictmem
python scripts/summarize_results.py
```

The GRU baseline uses 32-dimensional embeddings and 64 hidden units. The
sequence-adapted CTM receives each event exactly once, one event per recurrent
tick. Thus event gap is the actual temporal separation in its recurrence.
Vanilla CTM uses `NLM([recent M values])`. EvictMem-CTM adds one scalar per
neuron: `m_t = alpha*m_(t-1) + (1-alpha)*a_evicted`, then uses
`NLM([recent M values, m_t])`. Its memory starts at zero and updates only when
a sequence-generated pre-activation leaves the FIFO. Both CTMs use 32-dimensional
embeddings, 64 neurons, `memory_length=5`, `memory_hidden_dim=16`, and
`n_synch_out=64`; EvictMem uses fixed `alpha=0.95` by default.

All three models use AdamW at `1e-3`, batch size 128, up to 20 epochs, and
early stopping after 3 epochs without validation-F1 improvement. Best
checkpoints go to `checkpoints/` and raw metrics to `runs/`. Once all three
result files exist, the summary command writes `results/summary.csv`,
`results/gap_f1.csv`, and `results/f1_vs_gap.png` from test metrics. Use
matching `--seed` values for experiments and summarization (default 42).
