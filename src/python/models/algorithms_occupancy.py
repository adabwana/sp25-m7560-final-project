import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pyearth import Earth as MARS

# =============================================================================
# CUSTOM REGRESSOR WRAPPER FOR COUNT PREDICTIONS
# =============================================================================
class RoundedRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper for scikit-learn regressors that rounds predictions to the nearest integer.
    Ensures that all predictions are non-negative integers.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the wrapped estimator and round the predictions.
        
        Returns:
            np.ndarray: Rounded non-negative integer predictions.
        """
        y_pred = self.estimator_.predict(X)
        y_pred_rounded = np.round(y_pred).astype(int)
        y_pred_rounded = np.maximum(y_pred_rounded, 0)  # Ensure non-negative
        return y_pred_rounded

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def get_model_definitions():
    return {
        'MARS': (RoundedRegressor(Pipeline([
            ('mars', MARS())
        ])), {
            'model__estimator__mars__max_terms': [10, 20, 30],
            'model__estimator__mars__max_degree': [1, 2],
            'select_features__k': np.arange(70, 100, 10),
        }),
        'RandomForest': (RoundedRegressor(RandomForestRegressor()), {
            'model__estimator__n_estimators': [100, 200],
            'model__estimator__max_depth': [10, 20, None],
            'model__estimator__min_samples_split': [2, 5],
            'select_features__k': np.arange(70, 100, 10),
        }),
        'XGBoost': (RoundedRegressor(XGBRegressor()), {
            'model__estimator__n_estimators': [100, 200],
            'model__estimator__max_depth': [3, 6, 9],
            'model__estimator__learning_rate': [0.01, 0.1],
            'select_features__k': np.arange(70, 100, 10),
        }),
    } 