# EvictMem-CTM

GRU is the recurrent baseline. Sequence-adapted CTM processes one event per
recurrent tick. EvictMem-CTM adds compressed memory of evicted pre-activations
to the vanilla CTM.

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate evictmem-ctm

python -m unittest discover -s tests -v

python scripts/run_experiment.py --model gru
python scripts/run_experiment.py --model ctm
python scripts/run_experiment.py --model evictmem

python scripts/summarize_results.py
```

Raw run JSONs are saved in `runs/` and best checkpoints in `checkpoints/`.
The summarizer writes `summary.csv`, `gap_f1.csv`, and `f1_vs_gap.png` in
`results/`.
