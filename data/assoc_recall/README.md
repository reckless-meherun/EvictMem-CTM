# Associative-recall dataset

This synthetic dataset tests whether a model can retrieve a key's assigned value across a long sequence containing many competing assignments. It contains 20,000 samples: 14,000 training samples, 3,000 validation samples, and 3,000 test samples.

## Sample structure

Every sequence has 512 tokens and contains 24 distinct key-value assignments. An assignment is an adjacent `K<n> VALUE_<b>` pair, and each of its 24 keys occurs only once. Noise tokens fill all other positions except the last one. The final token is a query `Q<n>` asking for the value assigned earlier to `K<n>`; for example, `Q15` asks for the value paired with `K15`.

The vocabulary has 74 tokens:

- 32 keys: `K0` through `K31`
- 32 corresponding queries: `Q0` through `Q31`
- 2 values: `VALUE_0` and `VALUE_1`
- 8 noise tokens: `NOISE_0` through `NOISE_7`

The target assignment's value appears at one of five gaps from the final query: `{16, 64, 128, 256, 384}`. Here, `gap` is the number of positions from the target value token to the query token: `query_position - target_value_position`.

## NPZ fields

Each of `train.npz`, `val.npz`, and `test.npz` contains:

| Field | Meaning |
| --- | --- |
| `sequences` | Integer token IDs, shaped `[number_of_samples, 512]`. The ID-to-token mapping is in `metadata.json`. |
| `labels` | Correct binary value for each query: `0` means `VALUE_0`, and `1` means `VALUE_1`. |
| `gaps` | Distance from the queried key's value token to the final query token. |
| `query_key_ids` | ID of the queried key (`0`–`31`, corresponding to `K0`–`K31`). |
| `pair_ids` | Identifier shared by the two samples in a matched pair. Pair IDs are local to each split. |

`metadata.json` records the generation seed, sequence length, vocabulary size, allowed gaps, split sizes, and complete token-to-ID vocabulary.

## Readable example

A shortened illustration (the real sequence has 512 tokens and 24 assignments) is:

```text
NOISE_3 K15 VALUE_1 NOISE_0 K7 VALUE_0 ... NOISE_6 Q15
```

The final `Q15` asks for the earlier assignment to `K15`, so the correct value is `VALUE_1`. In an actual row, `gap` gives the exact distance between that `VALUE_1` token and `Q15`.

## Matched pairs

Samples are generated in matched positive/negative pairs. Both members have the same query key, gap, assignment locations, keys, noise, and overall token counts. They differ only by swapping `VALUE_0` and `VALUE_1` between the queried assignment and one decoy assignment, so their labels are opposite while their token-frequency histograms remain identical. A model therefore cannot solve the task by counting value tokens or exploiting simple global frequency differences; it must associate the queried key with the value at the correct location.

## Generate CSV previews

From the repository root, run:

```bash
python scripts/make_readable_assoc_recall.py
```

This reads the existing dataset without modifying it and writes the first 50 samples of each split to `train_preview.csv`, `val_preview.csv`, and `test_preview.csv` in this directory. The preview columns are `sample_id`, `query_key`, `correct_value`, `gap`, `pair_id`, and `sequence`.
