import pandas as pd
from pathlib import Path

def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """
    Converts a CSV file to Parquet format.

    Args:
        csv_path (str): Path to the input CSV file.
        parquet_path (str): Path to the output Parquet file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Save as Parquet
    df.to_parquet(parquet_path, index=False)

if __name__ == "__main__":
    # Define the input CSV file and output Parquet file
    csv_file = "credit_card_transactions.csv"  # Replace with your CSV file name
    parquet_file = Path(csv_file).with_suffix('.parquet')

    # Convert the CSV to Parquet
    convert_csv_to_parquet(csv_file, str(parquet_file))

    print(f"Converted {csv_file} to {parquet_file}")