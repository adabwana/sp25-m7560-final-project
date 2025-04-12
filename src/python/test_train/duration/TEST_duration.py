import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt

# Local imports
project_root = '/workspaces/sp25-m7560-final-project'
sys.path.append(project_root)

from src.python.utils.preprocess import prepare_data
from src.python.utils.visualization_utils import calculate_metrics, plot_prediction_analysis, save_visualization_results, plot_feature_importance_biplot

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')

target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

X, y = prepare_data(df, target, features_to_drop)

# Time series train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=3,
    shuffle=False  # Maintains chronological order
)

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
# Set up MLflow tracking URI and artifact locations
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
# os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/.mlruns"

# Add error handling for MLflow connection
def check_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Please ensure MLflow server is running with:")
        print("mlflow server --backend-store-uri sqlite:///workspace/mlflow.db " + 
              "--default-artifact-root file:///workspace/mlruns --host 0.0.0.0 --port 5000")
        return False

# Check MLflow connection before proceeding
if not check_mlflow_connection():
    sys.exit(1)

# =============================================================================
# MODEL TESTING
# =============================================================================
experiment_base = "Duration_Pred"
client = mlflow.tracking.MlflowClient()

# Get all experiments
experiments = client.search_experiments(
    filter_string=f"name LIKE '{experiment_base}/%'"
)

test_results = []
best_model_info = {'rmse': float('inf')}

# Create results and duration subdirectory if they don't exist
os.makedirs(f'{project_root}/results/duration', exist_ok=True)

# In the testing loop, store the best model's predictions
best_model_predictions = None
best_model_rmse = float('inf')

# Test each model and find the best one
for experiment in experiments:
    model_name = experiment.name.split('/')[-1]
    print(f"\nTesting models for {model_name}...")
    
    # Pipeline types we used in training
    pipeline_types = ['vanilla', 'interact_select', 'pca_lda']
    cv_methods = ['kfold', 'rolling', 'expanding']
    
    experiment_results = []
    
    for pipeline_type in pipeline_types:
        for cv_name in cv_methods:
            full_model_name = f"{model_name}_{pipeline_type}_{cv_name}"
            print(f"  Testing {full_model_name}...")
            
            try:
                # Try to load the registered model
                model = mlflow.sklearn.load_model(f"models:/{full_model_name}/latest")
                print(f"    Successfully loaded model")
                
                # Make predictions and evaluate
                try:
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)
                    
                    # Store results with pipeline type
                    result = {
                        'Model': model_name,
                        'Pipeline': pipeline_type,
                        'CV_Method': cv_name,
                        **metrics  # Unpack all metrics from calculate_metrics
                    }
                    experiment_results.append(result)
                    
                    print(f"    RMSE: {metrics['RMSE']:.4f}")
                    
                    # Create prediction analysis plots
                    fig = plot_prediction_analysis(y_test, y_pred, full_model_name)
                    plt.savefig(f'{project_root}/results/duration/prediction_analysis_{full_model_name}.png')
                    plt.close()
                    
                    # Create feature importance biplot for the best model
                    if metrics['RMSE'] < best_model_rmse:
                        best_model_rmse = metrics['RMSE']
                        best_model_predictions = y_pred
                        # Create biplot for the best model
                        plot_feature_importance_biplot(
                            X_test, 
                            y_test, 
                            y_pred, 
                            X_test.columns,  # feature names
                            f'{project_root}/results/duration'
                        )
                    
                except Exception as e:
                    print(f"    Error during prediction: {e}")
                    continue
                    
            except Exception as e:
                print(f"    Couldn't load model: {e}")
                continue
    
    # If we have results for this experiment, keep only top 3 based on RMSE
    if experiment_results:
        # Sort by RMSE and keep top 3
        top_3_results = sorted(experiment_results, key=lambda x: x['RMSE'])[:3]
        test_results.extend(top_3_results)

if not test_results:
    print("\nNo models were successfully tested!")
    sys.exit(1)

# Create results DataFrame
results_df = pd.DataFrame(test_results)
print("\nTop 3 Models per Experiment:")
print(results_df.sort_values('RMSE'))

# Use visualization_utils functions to save results and create plots
save_visualization_results(results_df, project_root)