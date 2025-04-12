import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, FunctionTransformer

def get_model_definitions():
    return {
        'Ridge': (Ridge(), {
            'model__alpha': np.logspace(0, 2, 10),
            'select_features__k': np.arange(10, 55, 5),
        }),
        'Lasso': (Lasso(), {
            'model__alpha': np.logspace(-2, 0, 10),
            'select_features__k': np.arange(10, 55, 5),
        }),
        # 'ElasticNet': (ElasticNet(), {
        #     'model__alpha': np.logspace(-3, -1, 10),
        #     'model__l1_ratio': np.linspace(0.1, 0.9, 5),
        #     'select_features__k': np.arange(10, 55, 5),
        # }),
        'PenalizedSplines': (Pipeline([
            ('spline', SplineTransformer()),
            ('ridge', Ridge())
        ]), {
            'model__spline__n_knots': [9, 11, 13, 15],
            'model__spline__degree': [3],
            'model__ridge__alpha': np.logspace(0, 2, 20),
            'select_features__k': np.arange(10, 55, 5),
        }),
        'KNN': (KNeighborsRegressor(), {
            'model__n_neighbors': np.arange(15, 22, 2), # Creates [15, 17, 19, 21]
            'model__weights': ['uniform', 'distance'],
            # 'model__metric': ['euclidean', 'manhattan'],
            'select_features__k': np.arange(10, 55, 5),
        }),
        'PenalizedLogNormal': (Pipeline([
            ('log_transform', FunctionTransformer(
                func=lambda x: np.log1p(np.clip(x, 1e-10, None)),  # clip to prevent log(0)
                inverse_func=lambda x: np.expm1(x)
            )),
            ('ridge', Ridge())
        ]), {
            'model__ridge__alpha': np.logspace(0, 2, 20),
            'select_features__k': np.arange(10, 55, 5),
        }),
        # 'PenalizedExponential': (Pipeline([
        #     ('exp_transform', FunctionTransformer(
        #         func=lambda x: 1/np.clip(x, 1e-10, None),  # inverse link (canonical for exponential)
        #         inverse_func=lambda x: 1/np.clip(x, 1e-10, None)
        #     )),
        #     ('ridge', Ridge())
        # ]), {
        #     'model__ridge__alpha': np.logspace(0, 2, 20),
        #     'select_features__k': np.arange(10, 55, 5),
        # }),
    } 