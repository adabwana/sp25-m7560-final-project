library(testthat)
library(workflows)
library(recipes)
library(parsnip)
library(tune)
library(yardstick)
library(tibble)
library(dplyr)
library(here)

# Source dependencies
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))
source(here::here("src/r/tuning/tuning.R"))
source(here::here("src/r/training/training.R"))
# Source the evaluation functions
source(here::here("src/r/evaluation/evaluation.R"))

context("Model Prediction and Evaluation")

# --- Remove Placeholder Functions ---
# make_predictions <- function(...) { ... removed ... }
# evaluate_model <- function(...) { ... removed ... }
# --- End Placeholder Functions ---


# --- Test Setup ---
# Reuse setup from training tests
sample_data_eval_small <- tibble::tribble(
    ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date,
    "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
    "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10",
    "S3", "10:45:00", "CompSci", 95, 1, 11.1, "Senior", "2024-01-12", "2024-01-08", "2024-05-10",
    "S4", "12:15:00", "Physics", 45, 1, 9.1, "Sophomore", "2024-01-13", "2024-01-08", "2026-05-10",
    "S5", "09:00:00", "Math", 70, 1, 8.5, "Junior", "2024-01-14", "2024-01-08", "2025-05-10"
)
sample_data_eval <- bind_rows(replicate(6, sample_data_eval_small, simplify = FALSE)) %>%
    mutate(Duration_In_Min = Duration_In_Min + rnorm(n(), 0, 5)) %>%
    mutate(Student_IDs = paste0(Student_IDs, "_", row_number()))

# Create and fit a workflow
target_duration <- "Duration_In_Min"
features_to_drop_py <- c("Student_IDs", "Semester", "Expected_Graduation", "Check_Out_Time", "Session_Length_Category")
eval_recipe <- create_recipe(sample_data_eval, target_duration, features_to_drop_py)
eval_spec <- mars_spec # Using MARS spec
eval_wf <- build_workflow(eval_recipe, eval_spec)
best_params_sample <- head(mars_grid, 1)
# Fit the model (using function from training.R)
fitted_wf_eval <- train_final_model(eval_wf, best_params_sample, sample_data_eval)

# Sample new data for prediction (use subset of eval data for test)
new_data_sample <- head(sample_data_eval, 5)

# Define metrics for evaluate_model tests
regression_metrics_test <- metric_set(rmse, rsq, mae)

# --- Tests for make_predictions ---
test_that("make_predictions returns a tibble with .pred column", {
    # Use actual function
    preds <- make_predictions(fitted_wf_eval, new_data_sample)
    expect_s3_class(preds, "tbl_df")
    expect_true(".pred" %in% names(preds))
    expect_equal(nrow(preds), nrow(new_data_sample))
})

test_that("make_predictions requires a fitted workflow", {
    # Use actual function with an unfitted workflow
    unfitted_wf <- build_workflow(eval_recipe, eval_spec)
    expect_error(make_predictions(unfitted_wf, new_data_sample), "must be a trained workflow object")
})

# --- Tests for evaluate_model ---
test_that("evaluate_model calculates metrics correctly", {
    # Create dummy predictions and actuals
    set.seed(456)
    n_eval <- 10
    dummy_preds <- tibble(.pred = runif(n_eval, 40, 140))
    dummy_actuals_df <- tibble(actual_value = runif(n_eval, 30, 150))
    truth_col_name <- "actual_value"
    metrics_to_check <- metric_set(rmse, rsq, mae) # Keep this definition

    # Use actual function
    metrics <- evaluate_model(dummy_preds, dummy_actuals_df, truth_col_name, metrics_to_check)

    expect_s3_class(metrics, "tbl_df")
    expect_named(metrics, c(".metric", ".estimator", ".estimate"))
    # Check if the number of rows matches the number of metrics requested
    # This implicitly checks if all metrics were calculated successfully.
    expect_equal(nrow(metrics), length(attr(metrics_to_check, "metrics")))
    # Old check trying to subset the metric_set object:
    # expect_true(all(metrics_to_check$metric %in% metrics$.metric))
})

test_that("evaluate_model handles input errors", {
    # Use actual function
    dummy_preds <- tibble(.pred = 1)
    dummy_actuals_df <- tibble(value = 1)
    truth_col_name <- "value"

    expect_error(
        evaluate_model(tibble(wrong_col = 1), dummy_actuals_df, truth_col_name, regression_metrics_test),
        "must be a data frame/tibble with a .pred column"
    )
    expect_error(
        evaluate_model(dummy_preds, "not dataframe", truth_col_name, regression_metrics_test),
        "`actuals_data` must be a data frame or tibble"
    )
    expect_error(
        evaluate_model(dummy_preds, dummy_actuals_df, "wrong_col_name", regression_metrics_test),
        "must be a valid column name in `actuals_data`"
    )
    expect_error(
        evaluate_model(dummy_preds, bind_rows(dummy_actuals_df, dummy_actuals_df), truth_col_name, regression_metrics_test),
        "must have the same number of rows"
    )
    expect_error(
        evaluate_model(dummy_preds, dummy_actuals_df, truth_col_name, "not metric set"),
        "must be a metric_set object from yardstick"
    )
})
