library(testthat)
library(tibble) # For is_tibble
library(here)

# Source the actual implementation file
# Ensure the path is correct relative to the test file location
source(here::here("src/r/utils/data_utils.R"))

context("Data Loading Utilities")

# Define expected filenames
train_filename <- "train_engineered.csv"
test_filename <- "test_engineered.csv"
non_existent_filename <- "non_existent_file.csv"

# Placeholder function definition until implemented
# load_data <- function(file_path) { ... removed ... }

test_that("load_data loads training data correctly", {
    # Load using the actual function
    train_data <- load_data(train_filename)
    expect_true(tibble::is_tibble(train_data))
    # --- IMPORTANT: Update with actual column names ---
    # expect_true("duration_minutes" %in% colnames(train_data))
    # expect_true("feature1" %in% colnames(train_data))
    # Add checks for specific known columns once provided
    expect_true("Duration_In_Min" %in% colnames(train_data))
    expect_true("Student_IDs" %in% colnames(train_data))
    expect_gt(nrow(train_data), 0)
    expect_gt(ncol(train_data), 1) # Expect more than one column

    # Remove the skip message
    # skip("Skipping test until load_data is implemented and uses real data.")
})

test_that("load_data loads testing data correctly", {
    # Load using the actual function
    test_data <- load_data(test_filename)
    expect_true(tibble::is_tibble(test_data))
    # --- IMPORTANT: Update with actual column names ---
    # expect_true("duration_minutes" %in% colnames(test_data))
    # expect_true("feature1" %in% colnames(test_data))
    # Add checks for specific known columns once provided
    expect_true("Duration_In_Min" %in% colnames(test_data))
    expect_true("Student_IDs" %in% colnames(test_data))
    expect_gt(nrow(test_data), 0)
    expect_gt(ncol(test_data), 1) # Expect more than one column

    # Remove the skip message
    # skip("Skipping test until load_data is implemented and uses real data.")
})

test_that("load_data handles non-existent files", {
    # Expect error using the actual function
    expect_error(load_data(non_existent_filename), "File not found at path:")
})

# Clean up placeholder function if needed (though testthat runs in isolated environments)
# rm(load_data, envir = .GlobalEnv)
