"""Export small, human-readable previews of the associative-recall dataset."""

import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "assoc_recall"
SPLITS = ("train", "val", "test")
PREVIEW_SIZE = 50
FIELDNAMES = ("sample_id", "query_key", "correct_value", "gap", "pair_id", "sequence")


def main() -> None:
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    id_to_token = {token_id: token for token, token_id in metadata["vocabulary"].items()}
    value_0_id = metadata["vocabulary"]["VALUE_0"]

    for split in SPLITS:
        with np.load(DATA_DIR / f"{split}.npz", allow_pickle=False) as archive:
            count = min(PREVIEW_SIZE, len(archive["sequences"]))
            rows = []
            for sample_id in range(count):
                token_ids = archive["sequences"][sample_id]
                query_key_id = int(archive["query_key_ids"][sample_id])
                label = int(archive["labels"][sample_id])
                rows.append({
                    "sample_id": sample_id,
                    "query_key": id_to_token[query_key_id],
                    "correct_value": id_to_token[value_0_id + label],
                    "gap": int(archive["gaps"][sample_id]),
                    "pair_id": int(archive["pair_ids"][sample_id]),
                    "sequence": " ".join(id_to_token[int(token_id)] for token_id in token_ids),
                })

        output_path = DATA_DIR / f"{split}_preview.csv"
        with output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {count} samples to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
