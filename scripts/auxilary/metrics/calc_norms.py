import argparse
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_basic import min_max_norm, quantile

def main():
    parser = argparse.ArgumentParser(description="Calculate normalizations and quantiles for specified columns.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--minimax", nargs='*', default=[], help="Columns to minimax norm")
    parser.add_argument("--quantile", nargs='*', default=[], help="Columns to quantile norm")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)

    for col in args.minimax:
        if col not in ds.column_names:
            raise ValueError(f"Column {col} must exist in the dataset.")
        print(f"Computing min-max normalization for {col}...")
        norm = min_max_norm(ds[col])
        ds = ds.add_column(f"{col}_minimax", norm)

    for col in args.quantile:
        if col not in ds.column_names:
            raise ValueError(f"Column {col} must exist in the dataset.")
        print(f"Computing quantile for {col}...")
        q = quantile(ds[col])
        ds = ds.add_column(f"{col}_quantile", q)

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Done!")

if __name__ == "__main__":
    main()
