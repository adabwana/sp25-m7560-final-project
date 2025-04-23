# After run_pipeline.R finishes, this script generates diagnostic plots for the best models

# Load libraries
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(fs))

# Load the plotting functions
source(here::here("src/r/graphing/diagnostic_plots_duration.R"))
source(here::here("src/r/graphing/diagnostic_plots_occupancy.R"))
source(here::here("src/r/utils/config_utils.R"))

# --- Load Configuration ---
cfg <- load_config()
MODELS_DIR <- get_config_value(cfg, "paths.models")

# Set up output directory for plots
plot_output_dir <- here::here("artifacts", "r", "plots")
dir.create(plot_output_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------
# Function to find the best model files for a target variable
# ---------------------------------------------------------

find_best_model <- function(target_var, metric = "rmse") {
  # List all model files in the models directory
  model_files <- fs::dir_ls(here::here(MODELS_DIR), glob = "*.rds")
  
  # Filter for files matching the target variable and containing "last_fit"
  target_files <- model_files[grepl(paste0("^", target_var), basename(model_files)) & 
                             grepl("last_fit", basename(model_files))]
  
  if(length(target_files) == 0) {
    cat(glue::glue("No model files found for target variable '{target_var}'.\n"))
    return(NULL)
  }
  
  # Find the one with the best metric
  # For rmse, lower is better, so we'd want the file with the lowest number
  # This assumes files are named with the pattern: [Target]_best_holdout_[metric]_[value]_[model]_last_fit.rds
  
  # Extract the metric values from filenames
  metric_values <- sapply(target_files, function(f) {
    # Extract the value part after the metric name
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
    cat(glue::glue("Could not determine best model for '{target_var}' with metric '{metric}'.\n"))
    return(NULL)
  }
  
  best_file <- target_files[best_idx]
  cat(glue::glue("Best {target_var} model found: {basename(best_file)}\n"))
  return(best_file)
}

# ---------------------------------------------------------
# Generate plots for Duration model
# ---------------------------------------------------------

cat("\n--- Generating plots for Duration model ---\n")
duration_model_path <- find_best_model("Duration_In_Min", "rmse")

if(!is.null(duration_model_path)) {
  # Load the model
  duration_last_fit_obj <- readRDS(duration_model_path)
  
  # Generate and save plots
  generate_diagnostic_plots_duration(
    last_fit_result = duration_last_fit_obj,
    target_var_name = "Duration_In_Min",
    output_dir = plot_output_dir,
    plot_filename = "holdout_diagnostics_duration.png"
  )
  
  cat(glue::glue("Duration model plots saved to {plot_output_dir}/holdout_diagnostics_duration.png\n"))
} else {
  cat("Skipping Duration model plots due to missing model files.\n")
}

# ---------------------------------------------------------
# Generate plots for Occupancy model
# ---------------------------------------------------------

cat("\n--- Generating plots for Occupancy model ---\n")
occupancy_model_path <- find_best_model("Occupancy", "rmse")

if(!is.null(occupancy_model_path)) {
  # Load the model
  occupancy_last_fit_obj <- readRDS(occupancy_model_path)
  
  # Generate and save plots
  generate_diagnostic_plots_occupancy(
    last_fit_result = occupancy_last_fit_obj,
    target_var_name = "Occupancy",
    output_dir = plot_output_dir,
    plot_filename = "holdout_diagnostics_occupancy.png"
  )
  
  cat(glue::glue("Occupancy model plots saved to {plot_output_dir}/holdout_diagnostics_occupancy.png\n"))
} else {
  cat("Skipping Occupancy model plots due to missing model files.\n")
}

cat("\n--- Plotting complete ---\n")
