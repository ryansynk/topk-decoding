import argparse
import os
from pathlib import Path
import polars as pl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="CSV file from running RULER"
    )
    return parser.parse_args()


def main():
    args = get_args()
    try:
        print(f"Loading: {args.file}")
        # Read the CSV file into a Polars DataFrame
        df = pl.read_csv(args.file)
    except pl.exceptions.ComputeError as e:
        print(f"Warning: Could not read '{args.file}' due to error: {e}")
        raise
    except Exception as e:
        print(
            f"An unexpected error occurred while reading '{args.file}': {e}"
        )
        raise

    grouped = df.group_by("decode_strategy").agg(
        (pl.col("total_time").sum() / pl.col("num_tokens").sum()).alias("time_per_token"),
        (pl.col("num_tokens").sum() / pl.col("total_time").sum()).alias("tokens_per_second"),
        pl.col("correct").sum() / df.height,
        pl.col("peak_gpu_memory").max(),
    )
    print(grouped)


if __name__ == "__main__":
    main()
