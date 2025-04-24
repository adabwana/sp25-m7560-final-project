import os
import sys
import pandas as pd

def csv_to_parquet(csv_path):
    # Ensure the processed directory exists
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Determine output file path
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    parquet_path = os.path.join(processed_dir, f"{base_name}.parquet")

    # Write to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet file to: {parquet_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parquet_convert.py <input_csv_path>")
        sys.exit(1)
    csv_to_parquet(sys.argv[1])
