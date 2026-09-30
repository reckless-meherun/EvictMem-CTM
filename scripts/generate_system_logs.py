"""Generate the synthetic system-log dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evictmem_ctm.data.system_logs import generate, save_dataset, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/system_logs"))
    args = parser.parse_args()
    splits = generate(args.seed)
    validate(splits, check_regeneration=False)
    save_dataset(args.output_dir, splits, args.seed)
    validate(splits, args.output_dir)
    print(f"Generated and validated {sum(len(split['labels']) for split in splits.values())} examples in {args.output_dir}")


if __name__ == "__main__":
    main()
