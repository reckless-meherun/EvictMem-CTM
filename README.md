# EvictMem-CTM

## Synthetic system logs

Generate the 20,000-example long-range classification dataset with NumPy:

```bash
python scripts/generate_system_logs.py --seed 42
python scripts/inspect_system_logs.py
python -m unittest discover -s tests -v
```

If NumPy is unavailable, create a virtual environment and run these commands with
`.venv/bin/python`. Use `--output-dir` and `--data-dir` to choose another location.
The input `sequences` contain only token IDs; labels, gaps, pattern IDs, and
negative types are separate arrays. Positions are zero-based and the gap is
`position(B) - position(A)`. Positive examples have `negative_types=-1`;
benign negatives have `pattern_ids=-1`. Other negative pattern IDs identify
the A endpoint's pattern, except B-only examples, where they identify B's pattern.
Negative types 0–4 mean reversed, mismatched, A-only, B-only, and benign.
For one-endpoint and benign negatives, the assigned gap is a generation stratum;
it cannot be reconstructed from the visible event tokens alone.
