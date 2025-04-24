import pandas as pd
from typing import Tuple, List

def preprocess_data(df: pd.DataFrame, target_col: str, cols_to_drop: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies preprocessing steps to the dataframe.

    Steps:
    1. Separate features (X) and target (y).
    2. Drop specified columns from X.
    3. Convert 'Check_In_Time' to minutes since midnight.
    4. Convert categorical features to dummy variables.
    5. (Placeholder) Apply scaling.
    6. (Placeholder) Apply feature selection.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target variable column.
        cols_to_drop (List[str]): A list of column names to drop (excluding the target).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features (X) and target (y).

    Raises:
        ValueError: If target_col is not found in df.
    """
    print("Starting preprocessing...")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.copy()
    
    # 1. Separate target
    y = X.pop(target_col)
    print(f"Separated target variable: {target_col}")

    # 2. Drop specified feature columns (if they exist)
    actual_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    if actual_cols_to_drop:
        X = X.drop(columns=actual_cols_to_drop)
        print(f"Dropped columns: {actual_cols_to_drop}")
    else:
        print("No specified columns to drop were found in the feature set.")

    # 3. Convert Check_In_Time to minutes
    if 'Check_In_Time' in X.columns:
        print("Converting Check_In_Time to minutes...")
        try:
            # Assuming format H:M:S or similar parsable by pd.to_datetime
            time_dt = pd.to_datetime(X['Check_In_Time'], format='%H:%M:%S')
            X['Check_In_Time_Minutes'] = time_dt.dt.hour * 60 + time_dt.dt.minute
            X = X.drop(columns=['Check_In_Time'])
            print("Created 'Check_In_Time_Minutes' and dropped original.")
        except ValueError as e:
            print(f"Warning: Could not parse Check_In_Time. Error: {e}. Skipping conversion.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred during Check_In_Time conversion: {e}. Skipping.")
    else:
        print("'Check_In_Time' column not found, skipping time conversion.")
        
    # 4. Create dummy variables for remaining (i.e., not dropped) object/category columns
    # Ensure we only select from columns currently in X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if not categorical_cols.empty:
        print(f"Creating dummy variables for: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False) # Match recipes default
        print(f"DataFrame shape after dummies: {X.shape}")
    else:
        print("No categorical columns found for dummy encoding.")

    # 5. Placeholder: Scaling (e.g., StandardScaler)
    # scaler = StandardScaler()
    # numeric_cols = X.select_dtypes(include=np.number).columns
    # if numeric_cols.any():
    #     X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print("Placeholder: Scaling would be applied here.")
    
    # 6. Placeholder: Feature Selection (e.g., SelectKBest or based on VIP)
    print("Placeholder: Feature selection would be applied here.")

    print(f"Preprocessing finished. Final X shape: {X.shape}, y length: {len(y)}")
    return X, y 