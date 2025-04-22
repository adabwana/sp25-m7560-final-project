# After run_pipeline.R finishes...

# Load the plotting function
source(here::here("src/r/graphing/diagnostic_plots.R"))
source(here::here("src/r/utils/config_utils.R"))

# Load the saved last_fit result for the best model
# (Construct the filename based on the pipeline output/config)
# --- Load Configuration ---
cfg <- load_config()
MODELS_DIR <- get_config_value(cfg, "paths.models")

best_model_last_fit_path <- here::here("artifacts/models/r/Duration_In_Min_best_holdout_rmse_60-104_RandomForest_last_fit.rds") # Replace with actual filename
last_fit_obj <- readRDS(best_model_last_fit_path)

# Or generate and view directly
diagnostic_plot_object <- generate_diagnostic_plots(last_fit_obj, "Duration_In_Min")
print(diagnostic_plot_object)

# Generate and save plots
plot_output_dir <- here::here("artifacts", "r", "plots", "Duration_In_Min_YourModelName") # Example output dir
dir.create(plot_output_dir, recursive = TRUE, showWarnings = FALSE)

generate_diagnostic_plots(
    last_fit_result = last_fit_obj,
    target_var_name = "Duration_In_Min",
    output_dir = plot_output_dir,
    plot_filename = "holdout_diagnostics_duration.jpg"
)
