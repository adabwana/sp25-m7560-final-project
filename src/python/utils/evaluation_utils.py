import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Create a comparison plot of model performances."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='rmse', hue='cv_method', data=results_df)
    plt.xticks(rotation=45)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    return plt.gcf()

def evaluate_model_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate a model on the test set."""
    test_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, test_pred)) 