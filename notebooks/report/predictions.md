# Predictions

This chapter describes the process used to train the final models on the complete training dataset and generate predictions for the unlabeled test dataset (`LC_test`). This workflow leverages the best model configurations identified during the cross-validation and evaluation phases (detailed in `evaluation.md`).

The process is managed by two primary R scripts:

1.  [`scripts/train_final_model.R`](https://github.com/adabwana/sp25-m7560-final-project/blob/master/scripts/train_final_model.R): Trains the chosen best model on the *entire* `LC_train` dataset (Fall 2016-Spring 2017) using the optimal hyperparameters found previously, and saves the final fitted model object.
2.  [`scripts/predict_on_test.R`](https://github.com/adabwana/sp25-m7560-final-project/blob/master/scripts/predict_on_test.R): Loads the final trained model, loads the `LC_test` dataset (Fall 2017-Spring 2018), applies the model to generate predictions, performs any necessary post-processing, and saves the predictions.

## Final Model Training (`train_final_model.R`)

This script ensures the final production model utilizes all available labeled data for maximum performance.

1.  **Identify Best Configuration**: The script first searches the model artifacts directory (`artifacts/models/r/`) to find the files corresponding to the best performing model for the specified `TARGET_VARIABLE` (e.g., "Occupancy") based on the primary tuning metric (`TUNING_METRIC`, e.g., "rmse"). It extracts the model type (e.g., "XGBoost") and loads the best hyperparameter set (`_best_params.rds`).

    ```r
    # Conceptual logic from train_final_model.R
    best_model_files <- find_best_model_files(TARGET_VARIABLE, TUNING_METRIC)
    best_params <- readRDS(best_model_files$params_file)
    model_type <- best_model_files$model_type
    ```

2.  **Load Full Training Data**: It loads the complete `train_engineered.csv` dataset.

    ```r
    # Conceptual logic from train_final_model.R
    full_data <- load_data(DATA_FILENAME) # Loads LC_train
    ```

3.  **Rebuild Workflow**: The script rebuilds the `tidymodels` workflow:
    *   Creates the preprocessing `recipe` using the full training data (`create_recipe` from `src/r/recipes/recipes.R`).
    *   Gets the appropriate `parsnip` model specification for the best `model_type` (e.g., `xgb_spec` from `src/r/models/models.R`).
    *   Combines the recipe and model spec into a `workflow` object.
    *   Finalizes the workflow using the loaded `best_params`.

    ```r
    # Conceptual logic from train_final_model.R
    recipe_obj <- create_recipe(full_data, TARGET_VARIABLE, FEATURES_TO_DROP)
    model_spec <- model_list_to_use[[model_type]]$spec # Get spec based on type
    workflow_obj <- build_workflow(recipe_obj, model_spec)
    final_workflow <- finalize_workflow(workflow_obj, best_params)
    ```

4.  **Fit Final Model**: The finalized workflow is trained (`fit()`) using the entire `full_data`.

    ```r
    # Conceptual logic from train_final_model.R
    final_fit <- fit(final_workflow, data = full_data)
    ```

5.  **Save Model**: The final fitted workflow object (`final_fit`) is saved to the artifacts directory with a timestamp and model type identifier. A copy is also saved with a `_latest.rds` suffix for easy access by the prediction script.

    ```r
    # Conceptual logic from train_final_model.R
    final_model_path <- file.path(MODEL_DIR, final_model_filename)
    saveRDS(final_fit, final_model_path)
    # ... code to create/update _latest.rds symlink/copy ...
    ```

## Prediction Generation (`predict_on_test.R`)

This script handles the application of the trained model to new, unlabeled data.

1.  **Load Final Model**: It locates and loads the appropriate final fitted model object (preferring the `_latest.rds` version) for the specified `TARGET_PREDICTION`.

    ```r
    # Conceptual logic from predict_on_test.R
    # ... logic to find model_path for _latest.rds or timestamped .rds ...
    best_model_object <- readRDS(model_path)
    ```

2.  **Load Test Data**: The script loads the `test_engineered.csv` dataset, which contains the features for the prediction period (Fall 2017-Spring 2018) but no target variable values.

    ```r
    # Conceptual logic from predict_on_test.R
    test_data <- readr::read_csv(TEST_DATA_PATH, show_col_types = FALSE)
    ```

3.  **Generate Predictions**: The `predict()` function is called on the loaded model object with the `test_data`. The `tidymodels` workflow automatically applies the same preprocessing steps (defined in the recipe embedded within the fitted workflow) to the test data before feeding it to the underlying model engine (e.g., XGBoost).

    ```r
    # Conceptual logic from predict_on_test.R
    # Assumes make_predictions uses predict() internally
    predictions_df <- predict(best_model_object, new_data = test_data)
    # Renames default .pred column if needed by make_predictions
    ```

4.  **Post-process Predictions**: Task-specific adjustments are made. For `Occupancy` predictions, values are rounded to the nearest integer and floored at a minimum of 1 (since occupancy cannot be less than 1).

    ```r
    # Conceptual logic from predict_on_test.R
    if (TARGET_PREDICTION == "Occupancy") {
      predictions_df <- predictions_df %>%
        mutate(
          .pred = round(.pred),
          .pred = pmax(1, .pred) # Ensure minimum of 1
        )
    }
    ```

5.  **Save Predictions**: The resulting predictions are saved to a CSV file in the `data/predictions/` directory, named convention includes the target variable, model type, and timestamp.

    ```r
    # Conceptual logic from predict_on_test.R
    output_df <- predictions_df %>%
      mutate(.pred = round(.pred, digits = 4))
    # ... logic to generate output_path ...
    readr::write_csv(output_df, output_path)
    ```

This R-based pipeline ensures that the final model is trained optimally on all available labeled data and that predictions are generated consistently using the same preprocessing steps learned during training.
