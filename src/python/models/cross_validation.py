from sklearn.model_selection import KFold, TimeSeriesSplit

def get_cv_methods(n_samples: int):
    n_splits = 10
    default_test_size = n_samples // (n_splits + 1)

    return {
        'kfold': KFold(
            n_splits=10, 
            shuffle=True, 
            random_state=3
        ),
        'rolling': TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=default_test_size * 5,
            test_size=default_test_size
        ),
        'expanding': TimeSeriesSplit(
            n_splits=n_splits,
            test_size=default_test_size
        )
    } 