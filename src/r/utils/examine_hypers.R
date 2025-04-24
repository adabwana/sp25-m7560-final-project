# Load necessary libraries
library(tune)
library(readr)
library(dplyr)
library(purrr) # For map_dfr

# Define the path to the models
models_path <- file.path("artifacts", "models", "r")

# --- Load Full Tuning Results and Show Best ---

# List the RDS files containing the full tuning results (assuming pattern 'final...latest.rds')
tune_result_files <- list.files(
    path = models_path,
    pattern = "final_.*_latest\\.rds$", # Match _final_MODEL_latest.rds (adjusted escaping)
    full.names = TRUE
)

# Function to load RDS and show best based on RMSE
show_best_rmse <- function(file_path) {
    tryCatch(
        {
            tune_results <- readRDS(file_path)
            model_name <- gsub(".*final_|_latest\\.rds$", "", basename(file_path)) # Extract model name
            target_var <- gsub("_final.*", "", basename(file_path)) # Extract target variable

            cat("\n--- Best Hyperparameters (from ", basename(file_path), ") ---\n", sep = "")
            cat("Target Variable:", target_var, "\n")
            cat("Model:", model_name, "\n")

            # Check if it's a tune result object
            if (inherits(tune_results, "tune_results")) {
                best_params <- tune_results %>%
                    show_best(metric = "rmse", n = 1) # Assuming RMSE was the metric

                print(best_params)
                return(best_params %>% mutate(target = target_var, model = model_name, source_file = basename(file_path)))
            } else {
                cat("File does not contain a 'tune_results' object.\n")
                return(NULL)
            }
        },
        error = function(e) {
            cat("\nError processing file:", basename(file_path), "\n")
            cat("Error message:", conditionMessage(e), "\n")
            return(NULL)
        }
    )
}

# Apply the function to each file and combine results
best_params_from_tuning <- map_dfr(tune_result_files, show_best_rmse)

# --- Load Pre-selected Best Parameters ---

# List the RDS files containing only the best parameters
best_param_files <- list.files(
    path = models_path,
    pattern = "_best_params\\.rds$", # Match _best_params.rds (adjusted escaping)
    full.names = TRUE
)

# Function to load and display best params
load_and_show_best_params <- function(file_path) {
    tryCatch(
        {
            best_params <- readRDS(file_path)
            file_base <- gsub("_best_params\\.rds$", "", basename(file_path))
            # Extract target and model name (adjust regex if naming convention differs)
            target_var <- sub("^(.*?)_best_holdout.*$", "\\1", file_base)
            model_name <- sub(".*_(.*?)", "\\1", file_base)

            cat("\n--- Pre-selected Best Hyperparameters (from ", basename(file_path), ") ---\n", sep = "")
            cat("Target Variable:", target_var, "\n")
            cat("Model:", model_name, "\n")
            print(best_params)

            # Attempt to create a standardized tibble/data frame if possible
            # This assumes 'best_params' is a named list or similar structure
            if (is.list(best_params) && !is.data.frame(best_params)) {
                param_df <- as_tibble(best_params) %>%
                    mutate(target = target_var, model = model_name, source_file = basename(file_path))
                return(param_df)
            } else if (is.data.frame(best_params)) {
                return(as_tibble(best_params) %>% mutate(target = target_var, model = model_name, source_file = basename(file_path)))
            } else {
                cat("Could not convert loaded parameters to a standard format.\n")
                return(NULL) # Or handle differently
            }
        },
        error = function(e) {
            cat("\nError processing file:", basename(file_path), "\n")
            cat("Error message:", conditionMessage(e), "\n")
            return(NULL)
        }
    )
}

# Apply the function and combine results
best_params_preselected <- map_dfr(best_param_files, load_and_show_best_params)


# --- Comparison (Optional) ---
cat("\n
--- Summary of Best Parameters Found ---
")
cat("
From Full Tuning Results (show_best):
")
print(best_params_from_tuning)

cat("
From Pre-selected Files (_best_params.rds):
")
print(best_params_preselected)


cat("\nScript finished examining hyperparameters.\n")
