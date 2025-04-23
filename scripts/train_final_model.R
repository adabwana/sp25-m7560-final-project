#!/usr/bin/env Rscript

# --- Load Libraries ---
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(workflows))
suppressPackageStartupMessages(library(parsnip))
suppressPackageStartupMessages(library(recipes))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(fs))

# --- Source Helper Functions ---
source(here::here("src/r/utils/config_utils.R"))
source(here::here("src/r/utils/data_utils.R"))
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))

# --- Load Configuration ---
cfg <- load_config()

# --- Configuration ---
TARGET_VARIABLE <- get_config_value(cfg, "data.target_variable")
DATA_FILENAME <- get_config_value(cfg, "data.filename")
FEATURES_TO_DROP <- get_config_value(cfg, "data.features_to_drop")
MODEL_DIR <- here::here(get_config_value(cfg, "paths.models"))
TUNING_METRIC <- get_config_value(cfg, "model.tuning_metric", "rmse")
SEED <- get_config_value(cfg, "model.seed", 42)

# Add the *other* target variable to the drop list
if (TARGET_VARIABLE == "Duration_In_Min") {
    FEATURES_TO_DROP <- c(FEATURES_TO_DROP, "Occupancy")
} else if (TARGET_VARIABLE == "Occupancy") {
    FEATURES_TO_DROP <- c(FEATURES_TO_DROP, "Duration_In_Min")
} else {
    warning(glue::glue("Unknown TARGET_VARIABLE: '{TARGET_VARIABLE}'. Not adding default other target to FEATURES_TO_DROP."))
}

# --- Set seed for reproducibility ---
set.seed(SEED)

# ---------------------------------------------------------
# Function to find the best model files and parameters for a target variable
# ---------------------------------------------------------

find_best_model_files <- function(target_var, metric = "rmse") {
  # List all model files in the models directory
  model_files <- dir_ls(MODEL_DIR, glob = "*.rds")
  
  # Find files matching the target variable
  target_files <- model_files[grepl(paste0("^", target_var), basename(model_files))]
  
  if(length(target_files) == 0) {
    stop(glue::glue("No model files found for target variable '{target_var}'."))
  }
  
  # Extract relevant files
  last_fit_files <- target_files[grepl("last_fit", basename(target_files))]
  params_files <- target_files[grepl("best_params", basename(target_files))]
  
  if(length(last_fit_files) == 0) {
    stop(glue::glue("No last_fit files found for target variable '{target_var}'."))
  }
  
  if(length(params_files) == 0) {
    stop(glue::glue("No best_params files found for target variable '{target_var}'."))
  }
  
  # Find the best model based on the metric
  # Extract metric values from filenames
  metric_values <- sapply(last_fit_files, function(f) {
    parts <- strsplit(basename(f), "_")[[1]]
    metric_idx <- which(parts == metric) + 1
    if(metric_idx <= length(parts)) {
      # Convert from string format like "2-046" to numeric 2.046
      as.numeric(gsub("-", ".", parts[metric_idx]))
    } else {
      NA
    }
  })
  
  # For rmse or mae, lower is better
  if(metric %in% c("rmse", "mae")) {
    best_idx <- which.min(metric_values)
  } else {
    # For rsq, higher is better
    best_idx <- which.max(metric_values)
  }
  
  if(length(best_idx) == 0 || is.na(best_idx)) {
    stop(glue::glue("Could not determine best model for '{target_var}' with metric '{metric}'."))
  }
  
  best_last_fit_file <- last_fit_files[best_idx]
  
  # Find the corresponding params file (should have same base name except for suffix)
  best_basename <- str_replace(basename(best_last_fit_file), "_last_fit\\.rds$", "")
  best_params_file <- params_files[grepl(best_basename, basename(params_files))]
  
  if(length(best_params_file) == 0) {
    stop(glue::glue("No matching params file found for last_fit file: {basename(best_last_fit_file)}"))
  }
  
  # Get model type from filename
  model_type <- str_extract(basename(best_last_fit_file), "(?<=_)[^_]+(?=_last_fit\\.rds$)")
  
  return(list(
    last_fit_file = best_last_fit_file,
    params_file = best_params_file[1],
    model_type = model_type
  ))
}

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

cat(glue::glue("=== Training Final Model for {TARGET_VARIABLE} ===\n\n"))

# --- 1. Find the best model and hyperparameters ---
cat("--- Finding best model files and hyperparameters ---\n")
best_model_files <- find_best_model_files(TARGET_VARIABLE, TUNING_METRIC)

cat(glue::glue("Best model type: {best_model_files$model_type}\n"))
cat(glue::glue("Last fit file: {basename(best_model_files$last_fit_file)}\n"))
cat(glue::glue("Params file: {basename(best_model_files$params_file)}\n\n"))

# --- 2. Load the best hyperparameters ---
cat("--- Loading best hyperparameters ---\n")
best_params <- readRDS(best_model_files$params_file)
print(best_params)
cat("\n")

# --- 3. Load the full dataset ---
cat(glue::glue("--- Loading Full Dataset ({DATA_FILENAME}) ---\n"))
full_data <- load_data(DATA_FILENAME)
cat(glue::glue("Full data rows: {nrow(full_data)}\n\n"))

# --- 4. Create recipe from full data ---
cat(glue::glue("--- Creating Recipe for {TARGET_VARIABLE} using full data ---\n"))
recipe_obj <- create_recipe(full_data, TARGET_VARIABLE, FEATURES_TO_DROP)
print(recipe_obj)
cat("\n")

# --- 5. Get the correct model specification based on model type ---
cat(glue::glue("--- Creating Model Specification for {best_model_files$model_type} ---\n"))

# Determine model list to use
model_list_to_use <- if (TARGET_VARIABLE == "Duration_In_Min") {
    model_list_duration
} else if (TARGET_VARIABLE == "Occupancy") {
    model_list_occupancy
} else {
    stop(glue::glue("Unknown TARGET_VARIABLE: {TARGET_VARIABLE}"))
}

# Get model spec
if (!(best_model_files$model_type %in% names(model_list_to_use))) {
  stop(glue::glue("Model type '{best_model_files$model_type}' not found in model list for {TARGET_VARIABLE}."))
}

model_spec <- model_list_to_use[[best_model_files$model_type]]$spec

# --- 6. Create workflow ---
cat("--- Building Workflow ---\n")
workflow_obj <- build_workflow(recipe_obj, model_spec)

# --- 7. Finalize workflow with best parameters ---
cat("--- Finalizing Workflow with Best Parameters ---\n")
final_workflow <- finalize_workflow(workflow_obj, best_params)

# --- 8. Fit the model on full data ---
cat("--- Training Final Model on Full Dataset ---\n")
start_time <- Sys.time()
final_fit <- fit(final_workflow, data = full_data)
end_time <- Sys.time()
training_time <- difftime(end_time, start_time, units = "mins")
cat(glue::glue("Training completed in {round(as.numeric(training_time), 2)} minutes\n\n"))

# --- 9. Save the final model ---
cat("--- Saving Final Production Model ---\n")
timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
final_model_filename <- glue::glue("{TARGET_VARIABLE}_final_{best_model_files$model_type}_{timestamp}.rds")
final_model_path <- file.path(MODEL_DIR, final_model_filename)

saveRDS(final_fit, final_model_path)
cat(glue::glue("Final model saved to: {final_model_path}\n\n"))

# --- 10. Save a symlink to the latest model ---
latest_model_filename <- glue::glue("{TARGET_VARIABLE}_final_{best_model_files$model_type}_latest.rds")
latest_model_path <- file.path(MODEL_DIR, latest_model_filename)

# Remove existing symlink if it exists
if (file.exists(latest_model_path)) {
  file.remove(latest_model_path)
}
file.copy(final_model_path, latest_model_path)
cat(glue::glue("Latest model symlink created at: {latest_model_path}\n\n"))

cat("=== Final Model Training Complete ===\n") 