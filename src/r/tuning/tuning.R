# Load necessary libraries
library(tune)
library(workflows)
library(recipes)
library(parsnip)
library(rsample)
library(yardstick)
library(dplyr)

# Optional: Set up parallel processing (doParallel) if needed for speed
# library(doParallel)
# registerDoParallel(cores = parallel::detectCores(logical = FALSE) - 1)


#' Tune Workflow Hyperparameters over a Grid
#'
#' Uses `tune::tune_grid` to evaluate a workflow over a grid of hyperparameters
#' across specified resampling folds.
#'
#' @param workflow A tidymodels workflow object containing a preprocessor (recipe)
#'   and a model specification with tunable parameters.
#' @param resamples An `rset` object containing resampling folds (e.g., from
#'   `rsample::vfold_cv`).
#' @param grid A data frame or tibble where rows are possible hyperparameter
#'   combinations and columns are hyperparameters to tune.
#' @param metrics A metric set created by `yardstick::metric_set`.
#' @param control A control object created by `tune::control_grid` or
#'   `tune::control_race` to customize the tuning process (e.g., verbosity,
#'   saving predictions).
#'
#' @return A tibble with class `tune_results` containing the tuning results.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming tune_wf, tune_folds, tune_grid, tune_metrics, tune_control are defined
#' tuning_results <- tune_model_grid(
#'     workflow = tune_wf,
#'     resamples = tune_folds,
#'     grid = tune_grid,
#'     metrics = tune_metrics,
#'     control = tune_control
#' )
#' print(tuning_results)
#' tune::autoplot(tuning_results)
#' }
tune_model_grid <- function(workflow, resamples, grid, metrics, control = control_grid()) {
    # Input validation (basic checks, tune_grid does more)
    stopifnot(inherits(workflow, "workflow"))
    stopifnot(inherits(resamples, "rset"))
    stopifnot(inherits(grid, "data.frame"))
    stopifnot(inherits(metrics, "metric_set"))
    stopifnot(inherits(control, "control_grid") || inherits(control, "control_race"))

    # Perform grid search
    tune_res <- tune::tune_grid(
        object = workflow,
        resamples = resamples,
        grid = grid,
        metrics = metrics,
        control = control
    )

    return(tune_res)
}


#' Select Best Hyperparameters from Tuning Results
#'
#' Uses `tune::select_best` to identify the best hyperparameter combination
#' based on a specified metric from the results of `tune_grid`.
#'
#' @param tune_results A `tune_results` object obtained from `tune_model_grid`
#'   (or `tune::tune_grid`).
#' @param metric_name A single string specifying the metric to optimize
#'   (e.g., "rmse", "roc_auc"). `tune` will minimize or maximize appropriately
#'   based on the metric's known direction (or `maximize` argument in `select_best`).
#'
#' @return A tibble containing the best hyperparameter combination found.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming tuning_results is available
#' best_rmse_params <- select_best_hyperparameters(tuning_results, "rmse")
#' print(best_rmse_params)
#'
#' best_rsq_params <- select_best_hyperparameters(tuning_results, "rsq") # select_best knows rsq is maximized
#' print(best_rsq_params)
#' }
select_best_hyperparameters <- function(tune_results, metric_name) {
    # Input validation
    stopifnot(inherits(tune_results, "tune_results"))
    stopifnot(is.character(metric_name), length(metric_name) == 1)

    # Check if metric exists in results
    available_metrics <- tune::collect_metrics(tune_results)$.metric
    if (!metric_name %in% available_metrics) {
        stop(paste(
            "Metric", shQuote(metric_name), "not found in tune_results. Available metrics:",
            paste(unique(available_metrics), collapse = ", ")
        ), call. = FALSE)
    }

    # Select the best parameters
    # select_best automatically knows whether to minimize (e.g., rmse) or
    # maximize (e.g., rsq, accuracy, roc_auc)
    best_params <- tune::select_best(tune_results, metric = metric_name)

    return(best_params)
}
