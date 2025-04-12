import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy import stats

# Set default style
plt.style.use('seaborn-whitegrid')

def check_linearity_independence(model):
    """Check for linearity between different variables"""
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted_vals, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add smoothed line
    sns.regplot(x=fitted_vals, y=residuals, scatter=False, 
                color='gray', line_kws={'linestyle': '--'})
    
    plt.title('Check for Independence of\nRandom Error and Linearity')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    return plt

def check_normality_qq(model):
    """Check for normality of random error"""
    residuals = model.resid
    
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot of Residuals")
    return plt

def check_homoscedasticity(model):
    """Check for zero mean and constant variance of random error"""
    fitted_vals = model.fittedvalues
    std_residuals = model.get_influence().resid_studentized_internal
    
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted_vals, np.sqrt(np.abs(std_residuals)))
    
    # Add smoothed line
    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(std_residuals)), 
                scatter=False, color='gray', line_kws={'linestyle': '--'})
    
    plt.axhline(y=np.mean(np.sqrt(np.abs(std_residuals))), 
                color='r', linestyle='--')
    
    plt.title('Scale-Location')
    plt.xlabel('Fitted Values')
    plt.ylabel('sqrt(abs(Standardized Residuals))')
    return plt

def check_independence(model, sort_var):
    """Check for independence of random error"""
    residuals = model.resid
    X = model.model.exog
    df = pd.DataFrame(X, columns=model.model.exog_names)
    df['.resid'] = residuals
    
    # Sort by specified variable
    sorted_data = df.sort_values(by=sort_var, ascending=False)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(sorted_data)), sorted_data['.resid'])
    
    # Add smoothed line
    sns.regplot(x=range(len(sorted_data)), y=sorted_data['.resid'], 
                scatter=False, color='gray', line_kws={'linestyle': '--'})
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Check for Independence\nResiduals sorted by {sort_var}')
    plt.xlabel('Row Numbers')
    plt.ylabel('Residuals')
    return plt

def check_observed_vs_predicted(model, response):
    """Plot observed vs predicted values"""
    fitted_vals = model.fittedvalues
    observed_vals = model.model.endog
    
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted_vals, observed_vals)
    
    # Add diagonal line
    min_val = min(min(fitted_vals), min(observed_vals))
    max_val = max(max(fitted_vals), max(observed_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add smoothed line
    sns.regplot(x=fitted_vals, y=observed_vals, scatter=False, 
                color='gray', line_kws={'linestyle': '--'})
    
    plt.title('Observed vs Predicted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Actual Values')
    return plt

def check_residuals_vs_leverage(model):
    """Plot residuals vs leverage"""
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    std_residuals = influence.resid_studentized_internal
    
    plt.figure(figsize=(8, 6))
    plt.scatter(leverage, std_residuals)
    
    # Add smoothed line
    sns.regplot(x=leverage, y=std_residuals, scatter=False, 
                color='gray', line_kws={'linestyle': '--'})
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Leverage')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    return plt 