import argparse
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_basic import min_max_norm

def main():
    parser = argparse.ArgumentParser(description="Calculate min-max normalizations for specified columns.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of columns.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if any(col not in ds.column_names for col in args.columns):
        raise ValueError(f"All columns in {args.columns} must exist in the dataset.")

    for col in args.columns:
        print(f"Computing min-max normalization for {col}...")
        norm = min_max_norm(ds[col])
        ds = ds.add_column(f"{col}_minmax_norm", norm)

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset
    )
    print("Done!")

if __name__ == "__main__":
    main()
