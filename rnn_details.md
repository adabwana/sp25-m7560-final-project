## RNN Model Comparison: Gruey (minGRU) vs. Mtls (minLSTM)

This document outlines the evolution of our RNN-based models for predicting targets from tabular data, comparing the `gruey` setup (using `minGRU`) and the `mtls` setup (using `minLSTM`).

### 1. Core RNN Components & Helper Functions

*   **`minGRU` (`gruey_architecture.py`):**
    *   This model utilized a custom `minGRU` layer, likely based on recent research simplifying GRU structures.
    *   It included a model-specific hyperparameter: `gru_expansion_factor`, which adjusted the internal dimensionality within the GRU layer.
    *   Initially, `minGRU` implementations (like the one apparently used by `minLSTM`) depended on helper functions (`g`, `log_g`, `heinsen_associative_scan_log`, etc.) expected to be in a `mingru.py` file.

*   **`minLSTM` (`mtls_architecture.py`):**
    *   This model uses a custom `minLSTM` layer, also likely based on simplified LSTM structures (e.g., the one from arXiv:2401.01201v1 referenced in the code).
    *   Similar to `minGRU`, it uses an `lstm_expansion_factor` to control internal dimensions.
    *   Crucially, the provided `minLSTM.py` code *also* depended on the same helper functions (`default`, `exists`, `g`, `log_g`, `heinsen_associative_scan_log`) that were previously associated with `mingru`.

*   **Shared Helpers (`rnn_helpers.py`):**
    *   To resolve import errors and centralize code, we created `src/python/utils/rnn_helpers.py`.
    *   This file now contains the implementations for shared functions like `default`, `exists`, `g` (identity function), `log_g` (softplus), and `heinsen_associative_scan_log` (parallel scan approximation).
    *   Both `minlstm.py` and any similar `minGRU` implementation would now import these helpers from `rnn_helpers.py`, eliminating the dependency on a specific `mingru.py` file for these common utilities.

### 2. Modeling Approach 1: Independent Row Processing (Original `gruey.py`, Initial `mtls.py`)

*   **Technique:** Each row of the tabular input data (after preprocessing and scaling) was treated as an independent sample.
*   **Implementation:** Inside the model's `forward` method, the input `x` (shape `(batch_size, num_features)`) was explicitly reshaped using `x = x.unsqueeze(1)` to create a shape of `(batch_size, 1, num_features)`.
*   **RNN Role:** The `minGRU` or `minLSTM` layer processed this input as a **sequence of length 1**.
*   **Implications:**
    *   The RNN does **not** learn temporal dependencies *between rows* or sequence patterns.
    *   The RNN layer essentially acts as a sophisticated **non-linear feature transformation** layer. Its internal gates and recurrent connections process all input features simultaneously (within that single "time step") before passing the result to the final output layer.
    *   **`sequence_length` is fixed at 1 and is not a tunable hyperparameter** in this setup.
    *   This uses a standard `TabularDataset`.

### 3. Modeling Approach 2: Sequential Row Processing (Current `mtls.py`)

*   **Rationale:** Implemented because the underlying data was confirmed to be sorted chronologically, allowing for meaningful sequence modeling.
*   **Technique:** Treats the dataset as a time series, where each row is a time step. The model learns from a sequence of past rows to predict the target for the final row in that sequence.
*   **Implementation:**
    *   A new `SequenceDataset` (`src/python/datasets/sequence_dataset.py`) was created. It takes the full feature/target arrays and a `sequence_length`. Its `__getitem__` method returns `(X_sequence, y_target)`, where `X_sequence` has shape `(sequence_length, num_features)` and `y_target` is the target for the *last* row in `X_sequence`.
    *   The `unsqueeze(1)` was removed from the model's `forward` method. The model now directly accepts input of shape `(batch_size, sequence_length, num_features)`.
    *   The model takes the output corresponding to the *last time step* (`lstm_out[:, -1, :]`) before the final dropout/output layer.
*   **Implications:**
    *   The RNN (`minLSTM` in this case) now learns patterns based on the **sequence of rows**.
    *   It directly models the **temporal dependencies** present in the time-sorted data.
    *   **`sequence_length` becomes a meaningful and tunable hyperparameter**, controlling how much history the model uses for prediction.
    *   Requires the data to be meaningfully sorted (e.g., chronologically).
    *   Requires the custom `SequenceDataset` for data loading.

### 4. Hyperparameter Tuning Context

*   Across all variations (`gruey.py`, `gruey_occupancy.py`, `mtls.py`), Optuna was used to tune common hyperparameters (learning rate, dropout, weight decay, batch size, hidden dimensions, number of layers) and model-specific ones (`gru_expansion`/`lstm_expansion`, activation function).
*   The tuning ranges were iteratively refined based on the results of previous Optuna studies for each specific model/target combination.
*   Reporting was switched from MSE to RMSE for better interpretability, while still optimizing based on the underlying MSE loss.

This evolution shows two distinct ways to apply RNN architectures to tabular data: either as a complex feature extractor on independent rows or as a true sequence model when temporal relationships between rows exist and are leveraged in the data preparation.