import argparse
import os
from pathlib import Path
import polars as pl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gradientai/Llama-3-8B-Instruct-1048k", type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to place output csvs into",
    )
    parser.add_argument(
        "--N",
        type=int,
        choices=[32768, 65536, 131072],
        required=True,
    )
    parser.add_argument("--task", type=str, default="niah_multikey_1")
    return parser.parse_args()


def main():
    args = get_args()
    model_name_suffix = args.model.split("/")[-1]
    output_path = os.path.join(
        args.output_dir, f"{args.task}/{model_name_suffix}/N_{args.N}/{args.task}"
    )
    path_obj = Path(output_path)

    # Validate if the provided path is a directory and exists
    if not path_obj.is_dir():
        print(f"Error: '{output_path}' is not a valid directory or does not exist.")

    print(f"Searching for CSV files in: {output_path}")

    # List to store individual Polars DataFrames
    dataframes = []
    found_files = 0

    # Iterate over all files ending with '.csv' in the directory
    # glob('*.csv') finds all files matching the pattern directly in the directory
    for csv_file in path_obj.glob("*.csv"):
        if csv_file.is_file():  # Ensure it's actually a file
            try:
                print(f"Loading: {csv_file.name}")
                # Read the CSV file into a Polars DataFrame
                df = pl.read_csv(csv_file)
                dataframes.append(df)
                found_files += 1
            except pl.exceptions.ComputeError as e:
                print(f"Warning: Could not read '{csv_file.name}' due to error: {e}")
            except Exception as e:
                print(
                    f"An unexpected error occurred while reading '{csv_file.name}': {e}"
                )

    if not dataframes:
        print("No CSV files found or successfully loaded in the specified directory.")

    df = pl.concat(dataframes, how="vertical_relaxed")
    num_rows = df.height / found_files
    grouped = df.group_by("decode_strategy").agg(
        (num_rows * 5 / pl.col("total_time").sum()).alias("tokens_per_second"),
        pl.col("correct").sum() / num_rows,
        pl.col("peak_gpu_memory").max(),
    )
    print(grouped)


if __name__ == "__main__":
    main()
