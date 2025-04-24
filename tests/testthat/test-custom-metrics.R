library(testthat)
library(yardstick)
library(dplyr)
library(here)

# --- Load Custom Metrics ---
# Ensure the metrics defined in utils are available in the test environment
source(here::here("src/r/utils/metrics_utils.R"))

# --- Sample Data ---
# Create a tibble for testing
set.seed(123) # for reproducibility if random estimates are needed later
sample_data <- tibble::tibble(
    truth = c(1L, 2L, 3L, 4L, 5L, 6L, NA, 8L), # Integer truth
    estimate = c(1.1, 1.8, 3.4, 4.0, 4.9, 6.2, 7.0, NA), # Continuous estimate
    # Expected rounded estimates: 1, 2, 3, 4, 5, 6, 7, NA
)

# Create a second tibble used in several tests below (move definition here)
sample_vec_data <- tibble::tibble(
    truth = c(1L, 2L, 3L, 4L, 5L, 6L, NA, 8L),
    estimate = c(1.1, 1.8, 2.9, 4.0, 5.1, 6.2, 7.0, NA)
    # Rounded:    1,   2,   3,   4,   5,   6,   7,  NA
    # Truth:      1,   2,   3,   4,   5,   6,  NA,   8
    # Valid Pairs (Truth, Rounded): (1,1), (2,2), (3,3), (4,4), (5,5), (6,6)
    # Diffs: all 0. -> RMSE=0, MAE=0, RSQ=1
)

# --- Basic Sanity Checks ---
test_that("Custom metric functions exist", {
    expect_true(exists("rmse_int"), info = "rmse_int function should exist")
    expect_true(exists("mae_int"), info = "mae_int function should exist")
    expect_true(exists("rsq_int"), info = "rsq_int function should exist")

    expect_true(is.function(rmse_int), info = "rmse_int should be a function")
    expect_true(is.function(mae_int), info = "mae_int should be a function")
    expect_true(is.function(rsq_int), info = "rsq_int should be a function")

    # Check if they inherit from the correct yardstick class
    expect_s3_class(rmse_int, "numeric_metric")
    expect_s3_class(mae_int, "numeric_metric")
    expect_s3_class(rsq_int, "numeric_metric")
})

# --- Tests for Vector Functions (_vec) ---
test_that("_vec functions calculate correctly (na_rm = TRUE)", {
    # Manually calculate expected values based on rounded estimates
    # truth    = c( 1,  2,  3,  4,  5,  6)
    # estimate = c(1.1, 1.8, 3.4, 4.0, 4.9, 6.2)
    # rounded  = c( 1,  2,  3,  4,  5,  6)
    # diff     = c( 0,  0,  0,  0,  0,  0)
    # sq_diff  = c( 0,  0,  0,  0,  0,  0) -> RMSE = 0
    # abs_diff = c( 0,  0,  0,  0,  0,  0) -> MAE = 0
    # Rsq should be 1 (perfect fit after rounding for these)

    # Note: Sample data was modified slightly to make manual calc easier
    expected_rmse_int <- 0
    expected_mae_int <- 0
    expected_rsq_int <- 1

    expect_equal(rmse_int_vec(sample_vec_data$truth, sample_vec_data$estimate, na_rm = TRUE), expected_rmse_int)
    expect_equal(mae_int_vec(sample_vec_data$truth, sample_vec_data$estimate, na_rm = TRUE), expected_mae_int)
    expect_equal(rsq_int_vec(sample_vec_data$truth, sample_vec_data$estimate, na_rm = TRUE), expected_rsq_int)
})

test_that("_vec functions handle na_rm = FALSE", {
    # Expect NA if any input is NA
    expect_true(is.na(rmse_int_vec(sample_data$truth, sample_data$estimate, na_rm = FALSE)))
    expect_true(is.na(mae_int_vec(sample_data$truth, sample_data$estimate, na_rm = FALSE)))
    expect_true(is.na(rsq_int_vec(sample_data$truth, sample_data$estimate, na_rm = FALSE)))
})

# --- Tests for Data Frame Methods ---
test_that("Data frame methods produce correct tibble output", {
    # Expected values from _vec tests (na_rm = TRUE)
    expected_rmse_int <- 0
    expected_mae_int <- 0
    expected_rsq_int <- 1

    # Test individual calls
    rmse_res <- rmse_int(sample_vec_data, truth = truth, estimate = estimate)
    mae_res <- mae_int(sample_vec_data, truth = truth, estimate = estimate)
    rsq_res <- rsq_int(sample_vec_data, truth = truth, estimate = estimate)

    expect_s3_class(rmse_res, "tbl_df")
    expect_equal(nrow(rmse_res), 1)
    expect_equal(colnames(rmse_res), c(".metric", ".estimator", ".estimate"))
    expect_equal(rmse_res$.metric, "rmse_int")
    expect_equal(rmse_res$.estimator, "standard")
    expect_equal(rmse_res$.estimate, expected_rmse_int)

    expect_s3_class(mae_res, "tbl_df")
    expect_equal(nrow(mae_res), 1)
    expect_equal(colnames(mae_res), c(".metric", ".estimator", ".estimate"))
    expect_equal(mae_res$.metric, "mae_int")
    expect_equal(mae_res$.estimator, "standard")
    expect_equal(mae_res$.estimate, expected_mae_int)

    expect_s3_class(rsq_res, "tbl_df")
    expect_equal(nrow(rsq_res), 1)
    expect_equal(colnames(rsq_res), c(".metric", ".estimator", ".estimate"))
    expect_equal(rsq_res$.metric, "rsq_int")
    expect_equal(rsq_res$.estimator, "standard")
    expect_equal(rsq_res$.estimate, expected_rsq_int)
})


test_that("Custom metrics work with metric_set()", {
    custom_metric_set <- metric_set(rmse_int, mae_int, rsq_int)

    results <- custom_metric_set(sample_vec_data, truth = truth, estimate = estimate)

    expect_s3_class(results, "tbl_df")
    expect_equal(nrow(results), 3)
    expect_equal(colnames(results), c(".metric", ".estimator", ".estimate"))

    # Check values from the metric_set result
    expect_equal(results$.estimate[results$.metric == "rmse_int"], 0)
    expect_equal(results$.estimate[results$.metric == "mae_int"], 0)
    expect_equal(results$.estimate[results$.metric == "rsq_int"], 1)
})

# --- Optional: Add tests for edge cases or error handling ---
# test_that("Handles zero variance truth", { ... })
# test_that("Errors on invalid input types", { ... })
