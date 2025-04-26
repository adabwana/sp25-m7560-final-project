# Model Tuning Overview

This document outlines the different modeling approaches and hyperparameter tuning strategies employed in this project. For details on the preprocessing steps applied before model training, please refer to the `preprocessing_pipelines.md` document.

## Framework Architecture

To tackle the prediction tasks for visit duration and building occupancy, we adopted a pragmatic, hybrid modeling strategy. Recognizing the strengths of different toolkits, we combined the R ecosystem for some tasks and Python for others, using the engineered features from our R pipeline as common ground.

- **R (`tidymodels`)**: This was our workhorse for setting up and tuning Multivariate Adaptive Regression Splines (MARS), Random Forest, and XGBoost models. The structured approach of `tidymodels` (specifically `parsnip` for defining models, `recipes` for preprocessing, and `tune` for optimization) streamlined the workflow for these established algorithms. Key scripts include `src/r/models/models.R` and `scripts/run_pipeline.R`.
- **Python (PyTorch)**: For exploring neural network approaches, we turned to Python and PyTorch. This gave us the flexibility to define custom architectures for a Multi-Layer Perceptron (MLP) and a Gated Recurrent Unit (GRU) network. The relevant code can be found in `src/python/tuning/gruey.py` (for GRU tuning via Optuna) and `notebooks/team/neuralnets.rmd` (for MLP exploration).

This blend allowed us to leverage familiar and effective tools for standard models while retaining the power to experiment with more complex neural network designs.

## Cross-Validation and Holdout Strategy

Ensuring our models generalize beyond the data they were trained on was paramount. We implemented a consistent validation process across both R and Python frameworks:

1.  **Initial Split**: We first took the `LC_train` dataset (containing the fully labeled and engineered data from Fall 2016 - Spring 2017) and set aside 25% as a final **holdout set**. This data was *never* used during model tuning or selection.
2.  **Cross-Validation**: The remaining 75% of the `LC_train` data became our training/validation set. We used **5-fold cross-validation** on *this* set to tune the hyperparameters for each model (MARS, RF, XGBoost, MLP, GRU).
3.  **Tuning**: Within the 5 folds, models were trained on 4 folds and evaluated on the remaining fold. Performance metrics (primarily RMSE) averaged across the folds helped us identify the best-performing hyperparameter configuration for each algorithm.
4.  **Final Training**: Once the best hyperparameters were chosen for a model, we retrained that model one last time using the *entire* 75% training/validation set.
5.  **Holdout Evaluation**: Finally, this retrained model was evaluated on the **25% holdout set** we initially set aside. This provided the final, unbiased assessment of how well the model was likely to perform on truly unseen data.

This multi-step process helps guard against overfitting and gives us more confidence in our final performance metrics.

![Cross-Validation Diagram (Conceptual)](../../presentation/images/modeling/model_building.jpg)
*(Note: Diagram illustrates the general concept. Our specific process used 5 folds and an initial 80/20 split for the holdout set.)*

## Core Algorithms & Tuning

### R `tidymodels` Framework (MARS, Random Forest, XGBoost)

This part of the pipeline leveraged the integrated tools within `tidymodels` for model specification and hyperparameter tuning.

**Algorithms & Tuning:**

1.  **Multivariate Adaptive Regression Splines (MARS):**
    - *Concept*: Builds a model using linear segments (hinge functions) to capture non-linearities.
    - *Specification (`parsnip`)*:
      ```r
      # From src/r/models/models.R
      mars_spec <- mars(
        mode = "regression", num_terms = tune(), prod_degree = tune()
      ) %>%
        set_engine("earth")
      ```
    - *Tuned Hyperparameters (`dials` grid)*:
        - `num_terms`: Number of hinge functions. Explored ranges like [7, 15] (duration) and [120, 130] (occupancy).
        - `prod_degree`: Maximum interaction degree between terms. Fixed at 1 (no interactions).
      ```r
      # Example grid for occupancy (from src/r/models/models.R)
      mars_grid_occ <- grid_regular(
        parameters(
          num_terms(range = c(120L, 130L)),
          prod_degree(range = c(1L, 1L))
        ),
        levels = c(num_terms = 10, prod_degree = 1)
      )
      ```

2.  **Random Forest:**
    - *Concept*: An ensemble of decision trees, reducing variance and improving robustness.
    - *Specification (`parsnip`)*:
      ```r
      # From src/r/models/models.R
      rf_spec <- rand_forest(
        mode = "regression", trees = tune(), min_n = tune(), mtry = tune()
      ) %>%
        set_engine("ranger")
      ```
    - *Tuned Hyperparameters (`dials` grid)*:
        - `trees`: Number of trees. Ranges like [300, 325] (duration) and [250, 350] (occupancy).
        - `min_n`: Min data points in a node for splitting. Ranges like [15, 25] (duration) and [2, 3] (occupancy).
        - `mtry`: Number of predictors sampled at each split. Ranges like [20, 25] (duration) and [40, 45] (occupancy).
      ```r
      # Example grid for occupancy (from src/r/models/models.R)
      rf_grid_occ <- grid_regular(
        parameters(
          trees(range = c(250L, 350L)),
          min_n(range = c(2L, 3L)),
          mtry(range = c(40L, 45L))
        ),
        levels = c(trees = 3, min_n = 2, mtry = 2)
      )
      ```

3.  **XGBoost:**
    - *Concept*: Gradient boosting machine that builds trees sequentially, correcting errors of prior trees.
    - *Specification (`parsnip`)*:
      ```r
      # From src/r/models/models.R
      xgb_spec <- boost_tree(
        mode = "regression", trees = tune(), tree_depth = tune(), learn_rate = tune(),
        min_n = tune(), mtry = tune()
      ) %>%
        set_engine("xgboost")
      ```
    - *Tuned Hyperparameters (`dials` grid)*:
        - `trees`: Number of boosting rounds. Ranges like [75, 100] (duration) and [350, 450] (occupancy).
        - `tree_depth`: Max depth per tree. Ranges like [15, 21] (duration) and [6, 8] (occupancy).
        - `learn_rate`: Learning rate. Fixed values explored (e.g., 0.05, 0.1).
        - `min_n`: Min data points in a node. Ranges like [10, 15] (duration) and [2, 3] (occupancy).
        - `mtry`: Predictors sampled per tree. Ranges like [12, 15] (duration) and [30, 35] (occupancy).
      ```r
      # Example grid for occupancy (from src/r/models/models.R)
      xgb_grid_occ <- grid_regular(
        parameters(
          trees(range = c(350L, 450L)),
          tree_depth(range = c(6L, 8L)),
          learn_rate(range = log10(c(0.1, 0.1))),
          min_n(range = c(2L, 3L)),
          mtry(range = c(30L, 35L))
        ),
        levels = c(trees = 3, tree_depth = 3, learn_rate = 1, min_n = 2, mtry = 2)
      )
      ```

### Python Framework (MLP, GRU)

For neural networks, we shifted to Python and PyTorch, performing preprocessing steps as outlined in the previous section, `Preproccessing Pipelines`.

**Algorithms & Tuning:**

1.  **Multi-Layer Perceptron (MLP):**
    - *Concept*: A standard feedforward neural network.
    - *Architecture Definition (`nn.Module`)*:
      ```python
      # Simplified from notebooks/team/neuralnets.rmd
      import torch
      import torch.nn as nn

      class SimpleNN(nn.Module):
          def __init__(self, input_dim, hidden_dims, output_dim, activation='ReLU', dropout=0.0):
              super(SimpleNN, self).__init__()
              # ... (logic to build sequential layers based on hidden_dims, activation, dropout)
              layers = []
              prev_dim = input_dim
              # ... loop to add nn.Linear, activation, nn.Dropout ...
              layers.append(nn.Linear(prev_dim, output_dim))
              self.model = nn.Sequential(*layers)

          def forward(self, x):
              return self.model(x)
      ```
    - *Tuned Hyperparameters (Example Search Space)*:
      ```python
      # Example from notebooks/team/neuralnets.rmd
      search_space = {
          "n_layers": [1, 2, 3],
          "n_units_l0": [12, 50, 100, 150], # Units in first hidden layer
          "n_units_l1": [12, 50, 100],      # Units in second (if n_layers >= 2)
          "n_units_l2": [12, 50],          # Units in third (if n_layers == 3)
          "activation": ["ReLU"],           # Activation function
          "learning_rate": [0.01],           # Optimizer learning rate
          "batch_size": [2048],             # Training batch size
          "dropout": [0, 0.2, 0.3],          # Dropout rate
          "weight_decay": [0, 1e-5, 1e-4]   # L2 regularization
      }
      ```

2.  **Gated Recurrent Unit (GRU) Network:**
    - *Concept*: A type of recurrent neural network suitable for sequence data, though applied here to tabular features, potentially capturing interactions differently than MLPs.
    - *Architecture Definition (`nn.Module`)*: A custom `GrueyModel` class was defined (in `src/python/models/gruey_architecture.py`), incorporating GRU layers potentially followed by linear layers.
    - *Tuned Hyperparameters (Optuna Search Space)*: Tuning was performed using Optuna (`src/python/tuning/gruey.py`).
      ```python
      # Snippet from objective function in src/python/tuning/gruey.py
      def objective(trial: optuna.trial.Trial, ...):
          # Tunable
          lr = trial.suggest_float("lr", 1e-3, 5e-3, log=True)
          dropout_rate = trial.suggest_float("dropout_rate", 0.25, 0.42)
          weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-5, log=True)
          gru_dim = trial.suggest_categorical("gru_dim", [128, 256, 512])
          num_layers = trial.suggest_int("num_layers", 1, 2)
          batch_size = trial.suggest_categorical("batch_size", [64, 128])
          gru_expansion = trial.suggest_float("gru_expansion", 0.5, 1.4)
          # Fixed
          activation_fn_name = "relu"
          # ... rest of objective function ...
          return best_val_loss
      ```

## Framework Integration

This combined R and Python approach allowed us to cast a wide net, evaluating traditional statistical models alongside more contemporary neural networks. The `tidymodels` framework offered efficient tuning for the former, while PyTorch provided the flexibility needed for the latter. Tracking experiments across both environments was facilitated by **_MLflow_**, allowing us to compare results and ultimately select the best overall models based on their performance on the holdout set, as detailed in the Evaluation section.
