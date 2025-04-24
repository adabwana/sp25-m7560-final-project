import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a specified file path (CSV or Parquet).

    Args:
        file_path (str): The full path to the data file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
        Exception: For other potential errors during loading.
    """
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        if df.empty:
            print("Warning: Loaded dataframe is empty.")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: File is empty: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {file_path}\n{e}")
        raise IOError(f"Could not read file: {file_path}. Reason: {e}")