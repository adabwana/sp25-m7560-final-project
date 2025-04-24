# tests/testthat/test-diagnostic-plots.R

library(testthat)
library(ggplot2)
library(patchwork)
library(tune)
library(recipes)
library(parsnip)
library(dplyr)
library(rsample)
library(workflows)

# Source the function to be tested
# Assuming test runs from the project root or testthat finds it relative to tests/testthat
# Adjust path if needed based on how tests are run
source(here::here("src/r/graphing/diagnostic_plots.R"))

# --- Helper to Create a Minimal last_fit Fixture ---
# This avoids needing a pre-saved RDS file for basic tests
# Uses mtcars for simplicity
create_minimal_last_fit <- function() {
    set.seed(123)
    split <- initial_split(mtcars, prop = 0.75)
    train_data <- training(split)
    test_data <- testing(split)

    rec <- recipe(mpg ~ cyl + disp, data = train_data) %>%
        step_normalize(all_numeric_predictors())

    lm_spec <- linear_reg() %>%
        set_engine("lm")

    wf <- workflow() %>%
        add_recipe(rec) %>%
        add_model(lm_spec)

    # Use last_fit directly for simplicity in fixture creation
    last_fit_result <- last_fit(wf, split = split)
    return(last_fit_result)
}

# --- Test Cases ---

test_that("generate_diagnostic_plots runs without error on valid input", {
    # skip_on_cran()
    last_fit_fixture <- create_minimal_last_fit()
    expect_no_error(generate_diagnostic_plots(last_fit_fixture, target_var_name = "mpg"))
})

test_that("generate_diagnostic_plots returns a ggplot/patchwork object when output_dir is NULL", {
    # skip_on_cran()
    last_fit_fixture <- create_minimal_last_fit()
    plot_obj <- generate_diagnostic_plots(last_fit_fixture, target_var_name = "mpg")
    expect_s3_class(plot_obj, "patchwork")
    expect_s3_class(plot_obj, "ggplot") # patchwork inherits ggplot
})

test_that("generate_diagnostic_plots saves a file when output_dir is provided", {
    # skip_on_cran()
    last_fit_fixture <- create_minimal_last_fit()

    # Create a temporary directory for the test output
    temp_dir <- tempdir()
    # Ensure it's clean if somehow it existed
    if (dir.exists(file.path(temp_dir, "test_plots"))) {
        unlink(file.path(temp_dir, "test_plots"), recursive = TRUE)
    }
    output_path <- file.path(temp_dir, "test_plots")
    plot_filename <- "test_diagnostics.png"
    full_file_path <- file.path(output_path, plot_filename)

    # Run the function to save the plot
    generate_diagnostic_plots(last_fit_fixture, target_var_name = "mpg", output_dir = output_path, plot_filename = plot_filename)

    # Check if the file was created
    expect_true(file.exists(full_file_path))

    # Clean up the created directory and file
    unlink(output_path, recursive = TRUE)
})

test_that("generate_diagnostic_plots validates input type", {
    # skip_on_cran()
    # Test with a non-last_fit object (e.g., a data frame)
    expect_error(
        generate_diagnostic_plots(mtcars, target_var_name = "mpg"),
        "Input must be a 'last_fit' object"
    )
})

test_that("generate_diagnostic_plots errors if target_var_name is missing", {
    last_fit_fixture <- create_minimal_last_fit()
    expect_error(
        generate_diagnostic_plots(last_fit_fixture),
        "Argument 'target_var_name' must be provided"
    )
})

test_that("generate_diagnostic_plots errors if target_var_name is not in predictions", {
    last_fit_fixture <- create_minimal_last_fit()
    expect_error(
        generate_diagnostic_plots(last_fit_fixture, target_var_name = "non_existent_column"),
        "The specified target variable 'non_existent_column' was not found"
    )
})

# Optional: Add more tests, e.g., checking plot layers if needed
