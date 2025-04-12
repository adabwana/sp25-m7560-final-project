import pandas as pd
import os

def load_and_analyze(filepath, dataset_name):
    """Load CSV and print grouped analysis for a given dataset."""
    df = pd.read_csv(filepath)
    print(f"\n{'='*20} {dataset_name} Analysis {'='*20}\n")
    
    # 1. Group by CV Method
    print("\n--- Grouping by CV Method ---")
    cv_groups = df.groupby('CV_Method')[['RMSE', 'R2']].agg(['mean', 'std']).round(4)
    cv_groups = cv_groups.sort_values(('RMSE', 'mean'))
    print(cv_groups)
    
    # 2. Group by Pipeline
    print("\n--- Grouping by Pipeline ---")
    pipeline_groups = df.groupby('Pipeline')[['RMSE', 'R2']].agg(['mean', 'std']).round(4)
    pipeline_groups = pipeline_groups.sort_values(('RMSE', 'mean'))
    print(pipeline_groups)
    
    # 3. Group by Model
    print("\n--- Grouping by Model ---")
    model_groups = df.groupby('Model')[['RMSE', 'R2']].agg(['mean', 'std']).round(4)
    model_groups = model_groups.sort_values(('RMSE', 'mean'))
    print(model_groups)
    
    # 4. Best Overall Combination
    print("\n--- Best Overall Combination ---")
    best_row = df.loc[df['RMSE'].idxmin()]
    print(f"Model: {best_row['Model']}")
    print(f"Pipeline: {best_row['Pipeline']}")
    print(f"CV Method: {best_row['CV_Method']}")
    print(f"RMSE: {best_row['RMSE']:.4f}")
    print(f"R2: {best_row['R2']:.4f}")

def main():
    # Analyze occupancy results
    occupancy_path = "results/occupancy/test_evaluation.csv"
    if os.path.exists(occupancy_path):
        load_and_analyze(occupancy_path, "Occupancy")
    
    # Analyze duration results
    duration_path = "results/duration/test_evaluation.csv"
    if os.path.exists(duration_path):
        load_and_analyze(duration_path, "Duration")

if __name__ == "__main__":
    main()