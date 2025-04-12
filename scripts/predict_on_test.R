#!/usr/bin/env Rscript

# --- Load Libraries ---
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(workflows))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(stringr))

# --- Configuration ---
# Specify which target variable's model you want to use for predictions
TARGET_PREDICTION <- "Duration_In_Min" # Or "Occupancy"

TEST_DATA_PATH <- here::here("data", "processed", "test_engineered.csv")
MODEL_ARTIFACT_DIR <- here::here("artifacts", "r", "models")
PREDICTION_OUTPUT_DIR <- here::here("data", "predictions")

# --- Source Helper Functions ---
# Only need the prediction function
source(here::here("src/r/evaluation/evaluation.R"))

# --- Main Prediction Logic ---

cat(glue::glue("--- Starting Prediction Script for Target: {TARGET_PREDICTION} ---\n\n"))

# 1. Find the saved model file for the target
# Assumes filename format: {TARGET_VARIABLE}_best_model_{MODEL_TYPE}.rds
model_file_pattern <- glue::glue("{TARGET_PREDICTION}_best_model_\\w+\\.rds")
model_files <- list.files(MODEL_ARTIFACT_DIR, pattern = model_file_pattern, full.names = TRUE)

if (length(model_files) == 0) {
  stop(glue::glue("No saved model file found for target '{TARGET_PREDICTION}' in {MODEL_ARTIFACT_DIR}"))
} else if (length(model_files) > 1) {
  warning(glue::glue("Multiple model files found for target '{TARGET_PREDICTION}'. Using the first one: {basename(model_files[1])}"))
  model_path <- model_files[1]
} else {
  model_path <- model_files[1]
}

cat(glue::glue("--- Loading model from: {model_path} ---\n\n"))
best_model_object <- readRDS(model_path)

# 2. Load Test Data
cat(glue::glue("--- Loading test data from: {TEST_DATA_PATH} ---\n\n"))
test_data <- readr::read_csv(TEST_DATA_PATH, show_col_types = FALSE)
cat(glue::glue("Test data rows: {nrow(test_data)}\n\n"))

# 3. Make Predictions
cat("--- Generating predictions ---\n\n")
predictions_df <- make_predictions(best_model_object, test_data)

# 4. Format Output (Optional - customize as needed)
# Often useful to include original identifiers if available
# Assuming 'Student_IDs' and 'Course_Name' etc. might be needed for joining later
# If the test data doesn't have these, adjust accordingly
# For now, just save the predictions
output_df <- predictions_df # Can add `bind_cols(test_data %>% select(ID_COLUMNS), predictions_df)` if needed

# 5. Save Predictions
dir.create(PREDICTION_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
output_filename <- glue::glue("{TARGET_PREDICTION}_predictions.csv")
output_path <- file.path(PREDICTION_OUTPUT_DIR, output_filename)

cat(glue::glue("--- Saving predictions to: {output_path} ---\n\n"))
readr::write_csv(output_df, output_path)

cat("--- Prediction Script Complete ---\n\n")
