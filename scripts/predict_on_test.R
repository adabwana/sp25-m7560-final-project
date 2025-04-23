#!/usr/bin/env Rscript

# --- Load Libraries ---
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(workflows))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(fs))

# --- Configuration ---
# Specify which target variable's model you want to use for predictions
TARGET_PREDICTION <- "Occupancy" # "Duration_In_Min" Or "Occupancy"

TEST_DATA_PATH <- here::here("data", "processed", "test_engineered.csv")
MODEL_ARTIFACT_DIR <- here::here("artifacts", "models", "r")
PREDICTION_OUTPUT_DIR <- here::here("data", "predictions")

# --- Source Helper Functions ---
# Only need the prediction function
source(here::here("src/r/evaluation/evaluation.R"))

# --- Main Prediction Logic ---

cat(glue::glue("--- Starting Prediction Script for Target: {TARGET_PREDICTION} ---\n\n"))

# Get all model files
all_model_files <- dir_ls(MODEL_ARTIFACT_DIR, glob = "*.rds")

# 1. Find the final model file for the target
# First try to find the "latest" model file which is a symlink
latest_model_pattern <- glue::glue("{TARGET_PREDICTION}_final_.*_latest\\.rds$")
latest_model_files <- all_model_files[grepl(latest_model_pattern, basename(all_model_files), perl = TRUE)]

if (length(latest_model_files) > 0) {
  # Use the latest model symlink
  model_path <- latest_model_files[1]
  cat(glue::glue("Using latest model: {basename(model_path)}\n"))
} else {
  # Fallback: Look for any final model file with timestamp
  final_model_pattern <- glue::glue("{TARGET_PREDICTION}_final_.*_[0-9]+\\.rds$")
  final_model_files <- all_model_files[grepl(final_model_pattern, basename(all_model_files), perl = TRUE)]
  
  if (length(final_model_files) > 0) {
    # Sort by filename (which should put the most recent timestamp last)
    final_model_files <- sort(final_model_files)
    model_path <- final_model_files[length(final_model_files)]
    cat(glue::glue("Using most recent final model: {basename(model_path)}\n"))
  } else {
    cat("No final models found. Looking for validation models instead...\n")
    # Last resort: Look for validation models from run_pipeline.R
    workflow_model_pattern <- glue::glue("{TARGET_PREDICTION}_best_.*_workflow\\.rds$")
    workflow_model_files <- all_model_files[grepl(workflow_model_pattern, basename(all_model_files), perl = TRUE)]
    
    if (length(workflow_model_files) > 0) {
      # Use the first available workflow model
      model_path <- workflow_model_files[1]
      cat(glue::glue("Using workflow model: {basename(model_path)}\n"))
    } else {
      stop(glue::glue("No model files found for target '{TARGET_PREDICTION}' in {MODEL_ARTIFACT_DIR}"))
    }
  }
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

# For Occupancy, round predictions to integers with minimum value of 1
if (TARGET_PREDICTION == "Occupancy") {
  cat("--- Rounding Occupancy predictions to integers (min=1) ---\n\n")
  predictions_df <- predictions_df %>%
    mutate(.pred = round(.pred),
           .pred = pmax(1, .pred))
}

# 4. Format Output
output_df <- predictions_df %>% 
    mutate(.pred = round(.pred, digits = 4))

# 5. Extract model type and generate filename
# Extract model type from the loaded model filename
model_basename <- basename(model_path)
model_type <- stringr::str_extract(model_basename, "(?<=_final_)[^_]+(?=_)")
if (is.na(model_type)) { # Fallback if pattern fails
    model_type <- stringr::str_extract(model_basename, "(?<=_)[^_]+(?=_workflow\\.rds$)")
    if (is.na(model_type)) { # Another fallback
        model_type <- "UnknownModel"
    }
}

# Generate timestamp
timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")

# Create prediction output directory
dir.create(PREDICTION_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Construct the filename
output_filename <- glue::glue("{TARGET_PREDICTION}_{model_type}_{timestamp}_pred.csv")
output_path <- file.path(PREDICTION_OUTPUT_DIR, output_filename)

cat(glue::glue("--- Saving predictions to: {output_path} ---\n\n"))
readr::write_csv(output_df, output_path)

cat("--- Prediction Script Complete ---\n\n")
