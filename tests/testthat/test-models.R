library(testthat)
library(parsnip)
library(dials)
library(tibble)
library(here)

# Source the model definitions file
source(here::here("src/r/models/models.R"))

context("Model Specification and Parameter Grids")

# --- MARS Tests ---
test_that("MARS specification is defined correctly", {
    expect_s3_class(mars_spec, "model_spec")
    expect_equal(mars_spec$mode, "regression")
    expect_equal(mars_spec$engine, "earth")
    # Check tunable parameters
    expect_true("num_terms" %in% names(mars_spec$args))
    expect_true("prod_degree" %in% names(mars_spec$args))
})

test_that("MARS parameter grid (duration) is correct", {
    expect_s3_class(mars_grid, "data.frame") # grid_regular produces tbl_df which inherits data.frame
    expect_named(mars_grid, c("num_terms", "prod_degree"))
    expect_equal(nrow(mars_grid), 3 * 2)
    expect_equal(sort(unique(mars_grid$num_terms)), c(10, 20, 30))
    expect_equal(sort(unique(mars_grid$prod_degree)), c(1, 2))
})

test_that("MARS parameter grid (occupancy) is correct", {
    expect_s3_class(mars_grid_occ, "data.frame") # Check inheritance
    expect_named(mars_grid_occ, c("num_terms", "prod_degree"))
    expect_equal(nrow(mars_grid_occ), 3 * 2)
    expect_equal(sort(unique(mars_grid_occ$num_terms)), c(10, 20, 30))
    expect_equal(sort(unique(mars_grid_occ$prod_degree)), c(1, 2))
    # In this case, they are identical, but good to test separately
    expect_equal(mars_grid, mars_grid_occ)
})

# --- Random Forest Tests ---
test_that("Random Forest specification is defined correctly", {
    expect_s3_class(rf_spec, "model_spec")
    expect_equal(rf_spec$mode, "regression")
    expect_equal(rf_spec$engine, "ranger")
    expect_true("trees" %in% names(rf_spec$args))
    expect_true("min_n" %in% names(rf_spec$args))
})

test_that("Random Forest parameter grid (duration) is correct", {
    expect_s3_class(rf_grid, "data.frame") # Check inheritance
    expect_named(rf_grid, c("trees", "min_n"))
    expect_equal(nrow(rf_grid), 2 * 2)
    expect_equal(sort(unique(rf_grid$trees)), c(100, 200))
    expect_equal(sort(unique(rf_grid$min_n)), c(2, 5))
})

test_that("Random Forest parameter grid (occupancy) is correct", {
    expect_s3_class(rf_grid_occ, "data.frame") # Check inheritance
    expect_named(rf_grid_occ, c("trees", "min_n"))
    expect_equal(nrow(rf_grid_occ), 2 * 2)
    expect_equal(sort(unique(rf_grid_occ$trees)), c(100, 200))
    expect_equal(sort(unique(rf_grid_occ$min_n)), c(2, 5))
    expect_equal(rf_grid, rf_grid_occ)
})

# --- XGBoost Tests ---
test_that("XGBoost specification is defined correctly", {
    expect_s3_class(xgb_spec, "model_spec")
    expect_equal(xgb_spec$mode, "regression")
    expect_equal(xgb_spec$engine, "xgboost")
    expect_true("trees" %in% names(xgb_spec$args))
    expect_true("tree_depth" %in% names(xgb_spec$args))
    expect_true("learn_rate" %in% names(xgb_spec$args))
})

test_that("XGBoost parameter grid (duration) is correct", {
    expect_s3_class(xgb_grid, "data.frame") # Check inheritance
    expect_named(xgb_grid, c("trees", "tree_depth", "learn_rate"))
    expect_equal(nrow(xgb_grid), 2 * 3 * 2)
    expect_equal(sort(unique(xgb_grid$trees)), c(100, 200))
    expect_equal(sort(unique(xgb_grid$tree_depth)), c(3, 6, 9))
    expect_equal(sort(unique(xgb_grid$learn_rate)), c(0.01, 0.1))
})

test_that("XGBoost parameter grid (occupancy) is correct", {
    expect_s3_class(xgb_grid_occ, "data.frame") # Check inheritance
    expect_named(xgb_grid_occ, c("trees", "tree_depth", "learn_rate"))
    expect_equal(nrow(xgb_grid_occ), 2 * 3 * 2)
    expect_equal(sort(unique(xgb_grid_occ$trees)), c(100, 200))
    expect_equal(sort(unique(xgb_grid_occ$tree_depth)), c(3, 6, 9))
    expect_equal(sort(unique(xgb_grid_occ$learn_rate)), c(0.01, 0.1))
    expect_equal(xgb_grid, xgb_grid_occ)
})

# --- List Tests ---
test_that("model_list_duration is structured correctly", {
    expect_type(model_list_duration, "list")
    expect_named(model_list_duration, c("MARS", "RandomForest", "XGBoost"))
    expect_s3_class(model_list_duration$MARS$spec, "model_spec")
    expect_s3_class(model_list_duration$MARS$grid, "data.frame") # Check inheritance
    expect_equal(model_list_duration$MARS$spec, mars_spec)
    expect_equal(model_list_duration$MARS$grid, mars_grid)
    # Add checks for RF and XGBoost if desired
})

test_that("model_list_occupancy is structured correctly", {
    expect_type(model_list_occupancy, "list")
    expect_named(model_list_occupancy, c("MARS", "RandomForest", "XGBoost"))
    expect_s3_class(model_list_occupancy$RandomForest$spec, "model_spec")
    expect_s3_class(model_list_occupancy$RandomForest$grid, "data.frame") # Check inheritance
    expect_equal(model_list_occupancy$RandomForest$spec, rf_spec)
    expect_equal(model_list_occupancy$RandomForest$grid, rf_grid_occ)
    # Add checks for MARS and XGBoost if desired
})
