# Load necessary libraries
library(workflows)
library(tune)
library(parsnip)
library(dplyr)

#' Train Final Model
#'
#' Finalizes a workflow with the best hyperparameters and fits it to the
#' entire training dataset.
#'
#' @param workflow A tidymodels workflow object containing a preprocessor (recipe)
#'   and a model specification with tunable parameters.
#' @param best_hyperparameters A tibble or data frame containing a single row
#'   with the best hyperparameter combination (e.g., from `select_best_hyperparameters`).
#' @param training_data The *entire* training dataset (data frame or tibble) to
#'   fit the final model on.
#'
#' @return A fitted workflow object.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming train_wf, best_params_sample, and sample_data_train are defined
#' final_fitted_model <- train_final_model(
#'     workflow = train_wf,
#'     best_hyperparameters = best_params_sample,
#'     training_data = sample_data_train
#' )
#' print(final_fitted_model)
#' # You can now use final_fitted_model for predictions
#' # predictions <- predict(final_fitted_model, new_data = test_data)
#' }
train_final_model <- function(workflow, best_hyperparameters, training_data) {
    # Input validation
    if (!inherits(workflow, "workflow")) {
        stop("`workflow` must be a workflow object.", call. = FALSE)
    }
    if (!inherits(best_hyperparameters, "data.frame") || nrow(best_hyperparameters) != 1) {
        stop("`best_hyperparameters` must be a data frame or tibble with exactly one row.", call. = FALSE)
    }
    if (!inherits(training_data, "data.frame")) {
        stop("`training_data` must be a data frame or tibble.", call. = FALSE)
    }

    # Finalize the workflow with the best hyperparameters
    final_wf <- tune::finalize_workflow(workflow, best_hyperparameters)

    # Fit the finalized workflow to the entire training dataset
    # Use last_fit() style approach? No, fit() is simpler here.
    # last_fit() is more for fitting on train and evaluating on test simultaneously.
    # We separate fitting and evaluation.
    final_fitted_wf <- parsnip::fit(final_wf, data = training_data)

    return(final_fitted_wf)
}
