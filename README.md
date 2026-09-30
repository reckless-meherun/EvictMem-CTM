# EvictMem-CTM

GRU is the recurrent baseline. Sequence-adapted CTM processes one event per
recurrent tick. EvictMem-CTM adds compressed memory of evicted pre-activations
to the vanilla CTM.

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate evictmem-ctm

python src/evictmem_ctm/data/assoc_recall.py
python -m unittest discover -s tests -v

python scripts/run_experiment.py --dataset assoc_recall --model gru
python scripts/run_experiment.py --dataset assoc_recall --model ctm
python scripts/run_experiment.py --dataset assoc_recall --model ctm_capacity
python scripts/run_experiment.py --dataset assoc_recall --model evictmem

python scripts/summarize_results.py --dataset assoc_recall
```

Associative recall defaults to 50 maximum epochs, 20 minimum epochs, patience
10, and validation accuracy for checkpoint selection. Its run JSONs, best
checkpoints, and summaries are saved in `runs/assoc_recall/`,
`checkpoints/assoc_recall/`, and `results/assoc_recall/`, respectively. The
summary files are `summary.csv`, `gap_f1.csv`, and `f1_vs_gap.png`.

The original `system_logs` benchmark remains available with its 20-epoch,
patience-3, validation-F1 policy.
