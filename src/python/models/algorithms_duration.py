import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pyearth import Earth as MARS

def get_model_definitions():
    return {
        'MARS': (Pipeline([
            ('mars', MARS())
        ]), {
            'model__mars__max_terms': [10, 20, 30],
            'model__mars__max_degree': [1, 2],
            'select_features__k': np.arange(10, 55, 5),
        }),
        'RandomForest': (RandomForestRegressor(), {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5],
            'select_features__k': np.arange(10, 55, 5),
        }),
        'XGBoost': (XGBRegressor(), {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6, 9],
            'model__learning_rate': [0.01, 0.1],
            'select_features__k': np.arange(10, 55, 5),
        }),
    } 