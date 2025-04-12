import pandas as pd
from typing import Tuple

# Global variables
target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

def get_feature_target_split(df: pd.DataFrame, target: str, features_to_drop: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.copy()
    
    # Filter features_to_drop to only include columns that exist
    missing_columns = [col for col in features_to_drop if col not in X.columns]
    existing_columns = [col for col in features_to_drop if col in X.columns]
    
    if missing_columns:
        print("\nWarning: The following columns were not found in the dataset:")
        for col in missing_columns:
            print(f"- {col}")
    
    # Drop existing columns
    if existing_columns:
        X = X.drop(columns=existing_columns, axis=1)
    
    # Get target if it exists
    y = df[target] if target in df.columns else None
    
    return X, y

def convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert date and time columns to appropriate formats."""
    X = X.copy()
    
    try:
        # Convert dates to datetime objects if columns exist
        datetime_columns = ['Check_In_Date', 'Semester_Date', 'Expected_Graduation_Date']
        existing_dates = [col for col in datetime_columns if col in X.columns]
        
        if existing_dates:
            for col in existing_dates:
                X[col] = pd.to_datetime(X[col], format='%Y-%m-%d')
        
        # Convert time to total minutes if column exists
        if 'Check_In_Time' in X.columns:
            X['Check_In_Time'] = pd.to_datetime(X['Check_In_Time'], format='%H:%M:%S').dt.hour * 60 + \
                                pd.to_datetime(X['Check_In_Time'], format='%H:%M:%S').dt.minute
        
        # Drop processed datetime columns
        X = X.drop(columns=existing_dates, axis=1, errors='ignore')
        
    except Exception as e:
        print(f"\nWarning: Error processing datetime features: {str(e)}")
    
    return X

def dummy(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical variables to dummies."""
    categorical_columns = X.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X

def prepare_data(df: pd.DataFrame, target: str, features_to_drop: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Pipeline to prepare data for modeling."""
    # Step 1: Split features and target
    X, y = get_feature_target_split(df, target, features_to_drop)
    
    # Step 2: Convert datetime features to numerical
    X = convert_datetime_features(X)
    
    # Step 3: Create dummies
    X = dummy(X)
    
    return X, y
