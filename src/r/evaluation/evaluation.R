# Load necessary libraries
library(workflows)
library(parsnip)
library(yardstick)
library(dplyr)
library(tibble)

#' Make Predictions using a Fitted Workflow
#'
#' Applies a fitted tidymodels workflow to new data to generate predictions.
#'
#' @param fitted_workflow A trained workflow object (output of `train_final_model` or `parsnip::fit`).
#' @param new_data A data frame or tibble containing the new data for which to
#'   generate predictions. Should have the same structure as the original training
#'   data *before* preprocessing.
#'
#' @return A tibble with prediction columns (e.g., `.pred` for regression).
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming final_fitted_model and test_data are defined
#' test_predictions <- make_predictions(final_fitted_model, test_data)
#' print(test_predictions)
#' }
make_predictions <- function(fitted_workflow, new_data) {
    # Input validation
    if (!inherits(fitted_workflow, "workflow") || !fitted_workflow$trained) {
        stop("`fitted_workflow` must be a trained workflow object.", call. = FALSE)
    }
    if (!inherits(new_data, "data.frame")) {
        stop("`new_data` must be a data frame or tibble.", call. = FALSE)
    }

    # Generate predictions
    predictions <- predict(fitted_workflow, new_data = new_data)

    return(predictions)
}


#' Evaluate Model Performance
#'
#' Calculates evaluation metrics by comparing model predictions to actual values.
#'
#' @param predictions A tibble containing the model predictions, typically with a
#'   `.pred` column (output of `make_predictions` or `predict`).
#' @param actuals_data A data frame or tibble containing the actual (true) outcome
#'   values. Must have the same number of rows as `predictions`.
#' @param truth_col A string specifying the column name in `actuals_data` that
#'   contains the true outcome values (e.g., "Duration_In_Min").
#' @param metrics_set A metric set object created by `yardstick::metric_set`
#'   (e.g., `metric_set(rmse, rsq, mae)`).
#'
#' @return A tibble containing the calculated metrics, with columns `.metric`,
#'   `.estimator`, and `.estimate`.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming test_predictions and test_data are defined
#' # Define the metrics we care about
#' regression_metrics <- metric_set(rmse, rsq, mae)
#'
#' # Calculate metrics
#' evaluation_results <- evaluate_model(
#'     predictions = test_predictions,
#'     actuals_data = test_data,
#'     truth_col = "Duration_In_Min", # Ensure this matches the actual column name
#'     metrics_set = regression_metrics
#' )
#' print(evaluation_results)
#' }
evaluate_model <- function(predictions, actuals_data, truth_col, metrics_set) {
    # Input validation
    if (!inherits(predictions, "data.frame") || !".pred" %in% names(predictions)) {
        stop("`predictions` must be a data frame/tibble with a .pred column.", call. = FALSE)
    }
    if (!inherits(actuals_data, "data.frame")) {
        stop("`actuals_data` must be a data frame or tibble.", call. = FALSE)
    }
    if (!is.character(truth_col) || length(truth_col) != 1 || !truth_col %in% names(actuals_data)) {
        stop(paste("`truth_col` must be a valid column name in `actuals_data`:", truth_col), call. = FALSE)
    }
    if (nrow(predictions) != nrow(actuals_data)) {
        stop("`predictions` and `actuals_data` must have the same number of rows.", call. = FALSE)
    }
    if (!inherits(metrics_set, "metric_set")) {
        stop("`metrics_set` must be a metric_set object from yardstick.", call. = FALSE)
    }

    # Combine predictions and truth for yardstick
    # Ensure the truth column is selected and renamed to 'truth'
    # Select .pred from predictions
    eval_data <- bind_cols(
        predictions %>% select(estimate = .pred),
        actuals_data %>% select(truth = all_of(truth_col))
    )

    # Calculate metrics
    metrics_result <- metrics_set(eval_data, truth = truth, estimate = estimate)

    return(metrics_result)
}
