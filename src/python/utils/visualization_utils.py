import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.dates as mdates
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_metrics(y_true, y_pred):
    """Calculate multiple regression metrics."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def plot_prediction_analysis(y_true, y_pred, model_name):
    """Create detailed prediction analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Prediction Analysis for {model_name}', fontsize=16)
    
    # Scatter plot of predicted vs actual values
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual Values')
    
    # Residuals plot
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30)
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    return fig

def plot_metric_comparisons(results_df: pd.DataFrame, save_dir: str):
    """Plot comparison of model performances for different metrics."""
    for metric in ['RMSE', 'MAE', 'R2']:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df,
            x='Model',
            y=metric,
            hue='Pipeline',
            palette='Set2'
        )
        plt.title(f'{metric} by Model and Pipeline Type (Top 3)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'top_3_{metric}_comparison.png'))
        plt.close()

def plot_parallel_coordinates(results_df: pd.DataFrame, save_dir: str):
    """
    Create a parallel coordinates plot to compare models across multiple metrics.
    This visualization helps identify trade-offs between different metrics.
    """
    # Create a copy of the dataframe for plotting
    plot_df = results_df.copy()
    
    # Normalize the metrics to 0-1 scale for better visualization
    metrics = ['RMSE', 'MAE', 'R2']
    for metric in metrics:
        if metric != 'R2':  # For RMSE and MAE, lower is better
            plot_df[metric] = (plot_df[metric] - plot_df[metric].min()) / (plot_df[metric].max() - plot_df[metric].min())
            plot_df[metric] = 1 - plot_df[metric]  # Invert so higher is better for all metrics
        else:  # For R2, higher is already better
            plot_df[metric] = (plot_df[metric] - plot_df[metric].min()) / (plot_df[metric].max() - plot_df[metric].min())
    
    # Create an index column for the parallel coordinates plot
    plot_df['Model_Config'] = plot_df['Model'] + '_' + plot_df['Pipeline'] + '_' + plot_df['CV_Method']
    
    # Create the parallel coordinates plot
    plt.figure(figsize=(12, 8))
    parallel_coordinates(plot_df, 'Model_Config', metrics, colormap=plt.cm.Set2)
    
    plt.title('Model Performance Comparison Across Metrics\n(Higher is Better for All Metrics)')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'parallel_coordinates_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_radar_comparison(results_df: pd.DataFrame, save_dir: str):
    """
    Create a radar/spider plot to compare the top models across different metrics.
    Each model is represented by a different colored polygon on the plot.
    """
    # Select metrics for comparison
    metrics = ['RMSE', 'MAE', 'R2']
    
    # Normalize the metrics to 0-1 scale
    plot_df = results_df.copy()
    for metric in metrics:
        if metric != 'R2':  # For RMSE and MAE, lower is better
            plot_df[metric] = (plot_df[metric] - plot_df[metric].min()) / (plot_df[metric].max() - plot_df[metric].min())
            plot_df[metric] = 1 - plot_df[metric]  # Invert so higher is better
        else:  # For R2, higher is already better
            plot_df[metric] = (plot_df[metric] - plot_df[metric].min()) / (plot_df[metric].max() - plot_df[metric].min())
    
    # Number of metrics
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot for each model
    for idx, row in plot_df.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        # Plot the model performance
        label = f"{row['Model']}_{row['Pipeline']}_{row['CV_Method']}"
        ax.plot(angles, values, 'o-', linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.title('Model Performance Comparison\n(Higher is Better for All Metrics)')
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'radar_plot_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance_biplot(X_test, y_test, y_pred, feature_names, save_dir: str):
    """
    Create both static and interactive 3D biplots showing the relationship between 
    top predictors and duration. Points are colored by actual duration values.
    """
    # Prepare the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Perform PCA with 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    exp_var_ratio = pca.explained_variance_ratio_
    
    # Get feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Calculate feature importance based on loading magnitudes
    feature_importance = np.sum(loadings**2, axis=1)
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    # Scale the arrows for visualization
    scaling_factor = np.abs(X_pca).max() / np.abs(loadings).max() * 0.7
    
    # ============= Create static matplotlib plots =============
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot samples, colored by actual duration
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=y_test, cmap='viridis',
                        alpha=0.6,
                        s=2)
    plt.colorbar(scatter, label='Study Duration (minutes)')
    
    # Plot feature vectors
    for idx in top_features_idx:
        x = loadings[idx, 0] * scaling_factor
        y = loadings[idx, 1] * scaling_factor
        z = loadings[idx, 2] * scaling_factor
        
        ax.quiver(0, 0, 0, x, y, z,
                 color='red', alpha=0.5,
                 arrow_length_ratio=0.15)
        
        ax.text(x * 1.15, y * 1.15, z * 1.15,
                feature_names[idx],
                color='red',
                fontsize=10,
                fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({exp_var_ratio[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({exp_var_ratio[1]:.1%} variance)')
    ax.set_zlabel(f'PC3 ({exp_var_ratio[2]:.1%} variance)')
    ax.set_title('3D Biplot: Feature Importance vs Study Duration')
    
    # Save multiple views of the static 3D plot
    views = [
        (20, 45),   # Default view
        (20, 135),  # Rotated 90 degrees
        (20, 225),  # Rotated 180 degrees
        (20, 315),  # Rotated 270 degrees
        (60, 45),   # Top-down view
        (0, 45)     # Side view
    ]
    
    for i, (elev, azim) in enumerate(views):
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'static_3d_biplot_view{i}.png'),
                    bbox_inches='tight', dpi=300)
    
    plt.close()
    
    # ============= Create interactive Plotly plot =============
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=y_test,
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Study Duration (minutes)')
        ),
        text=[f'Duration: {y:.1f} min<br>Predicted: {p:.1f} min' 
              for y, p in zip(y_test, y_pred)],
        hoverinfo='text',
        name='Data Points'
    ))
    
    # Add feature vectors as arrows
    for idx in top_features_idx:
        x = loadings[idx, 0] * scaling_factor
        y = loadings[idx, 1] * scaling_factor
        z = loadings[idx, 2] * scaling_factor
        
        fig.add_trace(go.Scatter3d(
            x=[0, x],
            y=[0, y],
            z=[0, z],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=['', feature_names[idx]],
            textposition='top center',
            textfont=dict(size=12, color='red'),
            hoverinfo='text',
            hovertext=f'{feature_names[idx]}<br>Importance: {feature_importance[idx]:.3f}',
            name=feature_names[idx]
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Interactive 3D Biplot: Feature Importance vs Study Duration',
            y=0.95
        ),
        scene=dict(
            xaxis_title=f'PC1 ({exp_var_ratio[0]:.1%} variance)',
            yaxis_title=f'PC2 ({exp_var_ratio[1]:.1%} variance)',
            zaxis_title=f'PC3 ({exp_var_ratio[2]:.1%} variance)',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        annotations=[
            dict(
                text=(f"Total explained variance: {sum(exp_var_ratio):.1%}<br>"
                      "Hover over points to see actual and predicted durations<br>"
                      "Drag to rotate, scroll to zoom"),
                xref="paper",
                yref="paper",
                x=0,
                y=0,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )
    
    # Save interactive plot
    fig.write_html(os.path.join(save_dir, 'interactive_3d_biplot.html'))

def save_visualization_results(results_df: pd.DataFrame, project_root: str):
    """Save all visualization results to the specified directory."""
    results_dir = os.path.join(project_root, 'results/duration')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results DataFrame
    results_df.to_csv(os.path.join(results_dir, 'top_models_comparison.csv'), index=False)
    
    # Create metric comparison plots
    plot_metric_comparisons(results_df, results_dir)
    
    # Create radar plot
    plot_radar_comparison(results_df, results_dir)
    
    # Print best model details
    best_result = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"\nBest Overall Model:")
    print(f"Model: {best_result['Model']}")
    print(f"Pipeline: {best_result['Pipeline']}")
    print(f"CV Method: {best_result['CV_Method']}")
    print(f"RMSE: {best_result['RMSE']:.4f}")
