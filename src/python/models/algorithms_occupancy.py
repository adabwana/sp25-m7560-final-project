import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.base import BaseEstimator, RegressorMixin, clone

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
        'Ridge': (RoundedRegressor(Ridge()), {
            'model__estimator__alpha': np.logspace(0, 2, 10),
            'select_features__k': np.arange(70, 100, 10),
        }),
        'Lasso': (RoundedRegressor(Lasso()), {
            'model__estimator__alpha': np.logspace(-2, 0, 10),
            'select_features__k': np.arange(70, 100, 10),
        }),
        # 'ElasticNet': (RoundedRegressor(ElasticNet()), {
        #     'model__estimator__alpha': np.logspace(-3, -1, 10),
        #     'model__estimator__l1_ratio': np.linspace(0.1, 0.9, 5),
        #     'select_features__k': np.arange(70, 100, 10),
        # }),
        'PenalizedSplines': (RoundedRegressor(Pipeline([
            ('spline', SplineTransformer()),
            ('ridge', Ridge())
        ])), {
            'model__estimator__spline__n_knots': [9, 11, 13, 15],
            'model__estimator__spline__degree': [3],
            'model__estimator__ridge__alpha': np.logspace(0, 2, 20),
            'select_features__k': np.arange(70, 100, 10),
        }),
        'KNN': (RoundedRegressor(KNeighborsRegressor()), {
            'model__estimator__n_neighbors': np.arange(15, 22, 2),  # Creates [15, 17, 19, 21]
            'model__estimator__weights': ['uniform', 'distance'],
            # 'model__estimator__metric': ['euclidean', 'manhattan'],
            'select_features__k': np.arange(70, 100, 10),
        }),
        'PenalizedPoisson': (RoundedRegressor(Pipeline([
            ('log_link', FunctionTransformer(
                func=lambda x: np.log(np.clip(x, 1e-10, None)),  # Log link (canonical for Poisson)
                inverse_func=lambda x: np.exp(np.clip(x, -10, 10))
            )),
            ('ridge', Ridge())
        ])), {
            'model__estimator__ridge__alpha': np.logspace(0, 2, 20),
            'select_features__k': np.arange(70, 100, 10),
        }),
        'PenalizedWeibull': (RoundedRegressor(Pipeline([
            ('weibull_link', FunctionTransformer(
                func=lambda x: np.log(-np.log(1 - np.clip(x / (x.max() + 1), 1e-10, 1-1e-10))),
                inverse_func=lambda x: (1 - np.exp(-np.exp(np.clip(x, -10, 10)))) * (x.max() + 1),
                check_inverse=False
            )),
            ('ridge', Ridge())
        ])), {
            'model__estimator__ridge__alpha': np.logspace(0, 2, 20),
            'select_features__k': np.arange(70, 100, 10),
        }),
    } 