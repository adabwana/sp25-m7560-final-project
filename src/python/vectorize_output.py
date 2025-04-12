import pandas as pd
import os

def combine_predictions():
    """Extract predictions into separate CSV files."""
    # Load predictions
    duration_path = 'results/predictions/duration_predictions.csv'
    occupancy_path = 'results/predictions/occupancy_predictions.csv'
    
    if not os.path.exists(duration_path) or not os.path.exists(occupancy_path):
        raise FileNotFoundError("Prediction files not found. Run predictions first.")
    
    # Read predictions
    duration_df = pd.read_csv(duration_path)
    occupancy_df = pd.read_csv(occupancy_path)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create and save duration predictions
    duration_output = pd.DataFrame({'Duration': duration_df['Predicted_Duration'].round(4)})
    duration_output.to_csv('output/duration.csv', index=False)
    
    # Create and save occupancy predictions
    occupancy_output = pd.DataFrame({'Occupancy': occupancy_df['Predicted_Occupancy']})
    occupancy_output.to_csv('output/occupancy.csv', index=False)
    
    print("\nPredictions Summary:")
    print(f"Number of predictions: {len(duration_df)}")
    print("\nDuration (minutes):")
    print(f"Mean: {duration_output['Duration'].mean():.2f}")
    print(f"Min: {duration_output['Duration'].min():.2f}")
    print(f"Max: {duration_output['Duration'].max():.2f}")
    print("\nOccupancy (students):")
    print(f"Mean: {occupancy_output['Occupancy'].mean():.1f}")
    print(f"Min: {occupancy_output['Occupancy'].min()}")
    print(f"Max: {occupancy_output['Occupancy'].max()}")
    print(f"\nPredictions saved to:")
    print(f"- output/duration.csv")
    print(f"- output/occupancy.csv")

if __name__ == "__main__":
    combine_predictions()
