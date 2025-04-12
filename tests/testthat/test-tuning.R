library(testthat)
library(tune)
library(workflows)
library(recipes)
library(parsnip)
library(rsample)
library(yardstick)
library(tibble)
library(dplyr)
library(here)

# Source dependencies
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))
# Source the tuning functions
source(here::here("src/r/tuning/tuning.R"))

context("Hyperparameter Tuning and Selection")

# --- Remove Placeholder Functions ---
# tune_model_grid <- function(...) { ... removed ... }
# select_best_hyperparameters <- function(...) { ... removed ... }
# --- End Placeholder Functions ---


# --- Test Setup ---
# Create a slightly larger sample dataset to avoid zero-variance issues in folds
sample_data_tune_small <- tibble::tribble(
    ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date,
    "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
    "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10",
    "S3", "10:45:00", "CompSci", 95, 1, 11.1, "Senior", "2024-01-12", "2024-01-08", "2024-05-10",
    "S4", "12:15:00", "Physics", 45, 1, 9.1, "Sophomore", "2024-01-13", "2024-01-08", "2026-05-10",
    "S5", "09:00:00", "Math", 70, 1, 8.5, "Junior", "2024-01-14", "2024-01-08", "2025-05-10"
)
# Replicate the data to increase size
sample_data_tune <- bind_rows(replicate(4, sample_data_tune_small, simplify = FALSE)) %>%
    # Add some noise to Duration_In_Min to ensure variance
    mutate(Duration_In_Min = Duration_In_Min + rnorm(n(), 0, 5)) %>%
    # Ensure Student IDs are unique if needed (though dropped in recipe)
    mutate(Student_IDs = paste0(Student_IDs, "_", row_number()))

target_duration <- "Duration_In_Min"
features_to_drop_py <- c("Student_IDs", "Semester", "Expected_Graduation", "Check_Out_Time", "Session_Length_Category")
tune_recipe <- create_recipe(sample_data_tune, target_duration, features_to_drop_py)
tune_spec <- mars_spec # Using MARS spec from models.R
tune_wf <- build_workflow(tune_recipe, tune_spec)

# Minimal grid for MARS
tune_grid <- head(mars_grid, 2) # Just use first 2 grid points for speed

# Resamples (keep v=2 for speed, but on larger data)
set.seed(123) # for reproducibility of folds
tune_folds <- rsample::vfold_cv(sample_data_tune, v = 2)

# Metrics set
tune_metrics <- metric_set(rmse, rsq, mae)

# Control object (important to suppress verbose output in tests)
tune_control <- control_grid(verbose = FALSE, save_pred = FALSE)


# --- Tests ---
test_that("tune_model_grid returns a tune_results object", {
    # Use actual function
    results <- tune_model_grid(tune_wf, tune_folds, tune_grid, tune_metrics, tune_control)

    expect_s3_class(results, "tune_results")
    expect_s3_class(results, "tbl_df") # It should also be a tibble

    # Check if results roughly match grid size (collect metrics first)
    collected <- tune::collect_metrics(results)
    expect_gt(nrow(collected), 0)
    expect_equal(length(unique(collected$.config)), nrow(tune_grid))
})

test_that("select_best_hyperparameters returns a tibble with correct parameters", {
    # Run the actual tuner first
    tune_results_real <- tune_model_grid(tune_wf, tune_folds, tune_grid, tune_metrics, tune_control)

    # Check that tuning actually produced some results before selecting
    collected_metrics_debug <- tune::collect_metrics(tune_results_real)
    expect_true(nrow(collected_metrics_debug) > 0,
        info = "Tune grid did not produce any metrics, cannot select best."
    )

    # --- Remove Debug: Print collected metrics ---
    # cat("\n--- Collected metrics inside test block: ---\n")
    # print(collected_metrics_debug)
    # cat("--- End collected metrics ---\n\n")
    # --- End Remove Debug ---

    # Select best using actual selector (RMSE)
    best_params_rmse <- select_best_hyperparameters(tune_results_real, "rmse")

    expect_s3_class(best_params_rmse, "tbl_df")
    expect_equal(nrow(best_params_rmse), 1)
    expect_true(all(c("num_terms", "prod_degree") %in% names(best_params_rmse))) # Check MARS params

    # Select based on rsq (higher is better)
    best_params_rsq <- select_best_hyperparameters(tune_results_real, "rsq")
    expect_s3_class(best_params_rsq, "tbl_df")
    expect_equal(nrow(best_params_rsq), 1)
    expect_true(all(c("num_terms", "prod_degree") %in% names(best_params_rsq))) # Check MARS params
})

test_that("select_best_hyperparameters handles invalid metric names", {
    # Run the actual tuner first
    tune_results_real <- tune_model_grid(tune_wf, tune_folds, tune_grid, tune_metrics, tune_control)

    # Check that tuning actually produced some results
    collected_metrics_debug_invalid <- tune::collect_metrics(tune_results_real)
    expect_true(nrow(collected_metrics_debug_invalid) > 0,
        info = "Tune grid did not produce any metrics, cannot test invalid metric selection."
    )

    # Test with invalid metric using actual selector
    expect_error(select_best_hyperparameters(tune_results_real, "invalid_metric"), "not found in tune_results")
})

# --- Test for RandomForest ---
test_that("tune_model_grid works for RandomForest", {
    # Setup specific to RF
    tune_spec_rf <- rf_spec # Using RF spec from models.R
    tune_grid_rf <- head(rf_grid, 2) # Use first 2 grid points for speed
    tune_wf_rf <- build_workflow(tune_recipe, tune_spec_rf)

    # Run the tuner
    results_rf <- NULL
    expect_no_warning( # Check for warnings like 'All models failed'
        results_rf <- tune_model_grid(tune_wf_rf, tune_folds, tune_grid_rf, tune_metrics, tune_control)
    )

    # Basic checks on results
    expect_s3_class(results_rf, "tune_results")
    collected_rf <- tune::collect_metrics(results_rf)
    expect_gt(nrow(collected_rf), 0)
    expect_equal(length(unique(collected_rf$.config)), nrow(tune_grid_rf))
})

# --- Test for XGBoost ---
test_that("tune_model_grid works for XGBoost", {
  # Setup specific to XGBoost
  tune_spec_xgb <- xgb_spec # Using XGB spec from models.R
  tune_grid_xgb <- head(xgb_grid, 2) # Use first 2 grid points for speed
  tune_wf_xgb <- build_workflow(tune_recipe, tune_spec_xgb)
  
  # Run the tuner
  results_xgb <- NULL
  expect_no_warning( # Check for warnings like 'All models failed'
    # Use tryCatch to potentially get more info on error if expect_no_warning fails
    tryCatch({ 
       results_xgb <- tune_model_grid(tune_wf_xgb, tune_folds, tune_grid_xgb, tune_metrics, tune_control)
    }, error = function(e) {
        message("Error during XGBoost tuning test: ", e$message)
        # Allow expect_no_warning to fail naturally
    })
  )
  
  # Basic checks on results (only if expect_no_warning passes)
  if (!is.null(results_xgb)) {
    expect_s3_class(results_xgb, "tune_results")
    collected_xgb <- tune::collect_metrics(results_xgb)
    expect_gt(nrow(collected_xgb), 0)
    expect_equal(length(unique(collected_xgb$.config)), nrow(tune_grid_xgb))
  } else {
    # If results are NULL due to caught error, fail explicitly
    fail("XGBoost tuning test failed to produce results.")
  }
})

# --- Adjust select_best tests to include XGBoost --- 
# (We can reuse the RF ones and just add an XGB version, or make them more generic)
# Let's add specific ones for XGB for clarity

test_that("select_best_hyperparameters returns a tibble with correct parameters (XGB)", {
    # Setup specific to XGBoost
    tune_spec_xgb <- xgb_spec 
    tune_grid_xgb <- head(xgb_grid, 2) 
    tune_wf_xgb <- build_workflow(tune_recipe, tune_spec_xgb)
    tune_results_xgb_real <- tune_model_grid(tune_wf_xgb, tune_folds, tune_grid_xgb, tune_metrics, tune_control)

    # Check that tuning actually produced some results before selecting
    collected_metrics_debug <- tune::collect_metrics(tune_results_xgb_real)
    expect_true(nrow(collected_metrics_debug) > 0, 
              info = "XGB Tune grid did not produce any metrics, cannot select best.")

    # Select best using actual selector (RMSE)
    best_params_rmse <- select_best_hyperparameters(tune_results_xgb_real, "rmse")
    
    expect_s3_class(best_params_rmse, "tbl_df")
    expect_equal(nrow(best_params_rmse), 1)
    expect_true(all(c("trees", "tree_depth", "learn_rate") %in% names(best_params_rmse))) # Check XGB params
    
    # Select based on rsq (higher is better) 
    best_params_rsq <- select_best_hyperparameters(tune_results_xgb_real, "rsq") 
    expect_s3_class(best_params_rsq, "tbl_df")
    expect_equal(nrow(best_params_rsq), 1)
    expect_true(all(c("trees", "tree_depth", "learn_rate") %in% names(best_params_rsq))) # Check XGB params
})

test_that("select_best_hyperparameters handles invalid metric names (XGB)", {
    # Setup specific to XGBoost
    tune_spec_xgb <- xgb_spec 
    tune_grid_xgb <- head(xgb_grid, 2) 
    tune_wf_xgb <- build_workflow(tune_recipe, tune_spec_xgb)
    tune_results_xgb_real <- tune_model_grid(tune_wf_xgb, tune_folds, tune_grid_xgb, tune_metrics, tune_control)

    # Check that tuning actually produced some results 
    collected_metrics_debug_invalid <- tune::collect_metrics(tune_results_xgb_real)
    expect_true(nrow(collected_metrics_debug_invalid) > 0, 
              info = "XGB Tune grid did not produce any metrics, cannot test invalid metric selection.")

    # Test with invalid metric using actual selector
    expect_error(select_best_hyperparameters(tune_results_xgb_real, "invalid_metric"), "not found in tune_results")
})

# ... (Keep previous RF-specific select_best tests as they are) ...
