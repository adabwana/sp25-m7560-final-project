#!/usr/bin/env Rscript

# --- Load Libraries ---
suppressPackageStartupMessages({
    library(here) # File path management
    library(dplyr) # Data manipulation
    library(rsample) # Data splitting & CV folds
    library(tune) # Hyperparameter tuning
    library(yardstick) # Model evaluation metrics
    library(workflows) # Combining recipes & models
    library(parsnip) # Model specifications
    library(recipes) # Preprocessing
    library(readr) # Reading/writing data
    library(glue) # Formatted string output
    library(argparse) # Command line argument parsing
    library(doParallel) # Parallel processing
})

# --- Source Helper Functions ---
source(here::here("src/r/utils/config_utils.R"))
source(here::here("src/r/utils/data_utils.R"))
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R")) # Defines model_list_duration, model_list_occupancy
source(here::here("src/r/workflows/workflows.R"))
source(here::here("src/r/tuning/tuning.R"))
source(here::here("src/r/training/training.R"))
source(here::here("src/r/evaluation/evaluation.R"))

# --- Argument Parsing ---
parser <- ArgumentParser(description = "Train and evaluate a single specified model.")
parser$add_argument("--target",
    type = "character", required = TRUE,
    help = "Target variable (e.g., 'Occupancy' or 'Duration_In_Min')"
)
parser$add_argument("--model",
    type = "character", required = TRUE,
    help = "Model name (must match a key in model_list_duration/occupancy)"
)

args <- parser$parse_args()

TARGET_VARIABLE <- args$target
MODEL_NAME <- args$model

cat(glue::glue("\n=== Running Single Model Pipeline ===\n"))
cat(glue::glue("Target Variable: {TARGET_VARIABLE}\n"))
cat(glue::glue("Model Name:      {MODEL_NAME}\n"))
cat(glue::glue("====================================\n\n"))

# --- Load Configuration ---
cfg <- load_config() # Assumes config files are in ./config relative to project root

# --- Configuration Values ---
DATA_FILENAME <- get_config_value(cfg, "data.filename", "train_engineered.csv")
TEST_SPLIT_PROP <- get_config_value(cfg, "data.test_split_prop", 0.8)
CV_FOLDS <- get_config_value(cfg, "model.cv_folds", 5)
SEED <- get_config_value(cfg, "model.seed", 3)
TUNING_METRIC <- get_config_value(cfg, "model.tuning_metric", "rmse")
FEATURES_TO_DROP <- get_config_value(cfg, "data.features_to_drop")
PARALLEL_ENABLED <- get_config_value(cfg, "parallel.enabled", TRUE)
NUM_CORES_TO_USE <- get_config_value(cfg, "parallel.num_cores", 6)
RESULTS_BASE_DIR <- here::here(get_config_value(cfg, "paths.results", "artifacts/params/r"))

# --- Validate Target Variable ---
if (!TARGET_VARIABLE %in% c("Occupancy", "Duration_In_Min")) {
    stop(glue::glue("Invalid --target: '{TARGET_VARIABLE}'. Must be 'Occupancy' or 'Duration_In_Min'."))
}

# Add the *other* target variable to the drop list
other_target <- if (TARGET_VARIABLE == "Duration_In_Min") "Occupancy" else "Duration_In_Min"
FEATURES_TO_DROP <- unique(c(FEATURES_TO_DROP, other_target))

# --- Parallel Processing Setup ---
cl <- NULL
if (PARALLEL_ENABLED) {
    cat(glue::glue("\n--- Registering parallel backend with {NUM_CORES_TO_USE} cores ---\n"))
    cl <- makePSOCKcluster(NUM_CORES_TO_USE)
    registerDoParallel(cl)
} else {
    cat("\n--- Parallel processing disabled ---\n")
    registerDoSEQ() # Ensure sequential backend is registered
}

# --- Determine Model List & Specific Model ---
model_list_to_use <- if (TARGET_VARIABLE == "Duration_In_Min") {
    model_list_duration
} else {
    model_list_occupancy
}

if (!MODEL_NAME %in% names(model_list_to_use)) {
    stop(glue::glue("Model '{MODEL_NAME}' not found in the model list for target '{TARGET_VARIABLE}'. Check src/r/models/models.R"))
}

model_info <- model_list_to_use[[MODEL_NAME]]
model_spec <- model_info$spec
model_grid <- model_info$grid

cat(glue::glue("Using model spec: {class(model_spec)[1]}\n"))
cat(glue::glue("Using grid size: {if(is.data.frame(model_grid)) nrow(model_grid) else 'N/A'}\n\n"))

# --- Define metric set based on task ---
# TODO: Make this dynamic based on target type if classification is added later
MODEL_METRICS <- metric_set(rmse, rsq, mae)

# --- Setup Output Directories ---
MODEL_RESULTS_DIR <- file.path(RESULTS_BASE_DIR, TARGET_VARIABLE, MODEL_NAME)
dir.create(MODEL_RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
cat(glue::glue("Results will be saved to: {MODEL_RESULTS_DIR}\n\n"))

# --- Pipeline Execution ---
set.seed(SEED)

# 1. Load Data
cat(glue::glue("--- Loading Data ({DATA_FILENAME}) ---\n"))
full_data <- load_data(DATA_FILENAME)
cat(glue::glue("Full data rows: {nrow(full_data)}\n\n"))

# 2. Split Data into Training and Testing
cat(glue::glue("--- Splitting Data (Train: {TEST_SPLIT_PROP*100}%, Test: {(1-TEST_SPLIT_PROP)*100}%) ---\n"))
data_split <- rsample::initial_split(full_data, prop = TEST_SPLIT_PROP, strata = NULL) # Add strata if needed
train_data <- rsample::training(data_split)
test_data <- rsample::testing(data_split)
cat(glue::glue("Training data rows: {nrow(train_data)}, Test data rows: {nrow(test_data)}\n\n"))

# 3. Create Recipe
cat(glue::glue("--- Creating Recipe for Target: {TARGET_VARIABLE} ---\n"))
recipe_obj <- create_recipe(train_data, TARGET_VARIABLE, FEATURES_TO_DROP)
print(recipe_obj)

# 4. Setup Resampling (Cross-Validation) for Tuning
cat(glue::glue("--- Setting up {CV_FOLDS}-fold Cross-Validation on Training Data ---\n"))
cv_folds <- rsample::vfold_cv(train_data, v = CV_FOLDS, strata = NULL) # Add strata if useful
# print(cv_folds)

# 5. Build Workflow
cat(glue::glue("--- Building Workflow ---\n"))
workflow_obj <- build_workflow(recipe_obj, model_spec)
# print(workflow_obj)

# 6. Tune Hyperparameters
cat(glue::glue("--- Tuning Hyperparameters ({MODEL_NAME}) ---\n"))
tune_control <- control_grid(verbose = FALSE, save_pred = FALSE, parallel_over = "everything")

tuning_results <- tryCatch(
    {
        tune_model_grid(
            workflow = workflow_obj,
            resamples = cv_folds,
            grid = model_grid,
            metrics = MODEL_METRICS,
            control = tune_control
        )
    },
    error = function(e) {
        warning(glue::glue("Tuning failed for model {MODEL_NAME}: {e$message}"), call. = FALSE)
        NULL # Return NULL if tuning fails
    }
)

# --- Save Tuning Results ---
if (!is.null(tuning_results)) {
    tuning_rds_path <- file.path(MODEL_RESULTS_DIR, "tuning_results.rds")
    cat(glue::glue("--- Saving Tuning Results to: {tuning_rds_path} ---\n"))
    saveRDS(tuning_results, tuning_rds_path)
    print(show_best(tuning_results, metric = TUNING_METRIC, n = 5))
} else {
    cat("--- Skipping model processing due to tuning failure ---\n")
    # Stop parallel backend if it was started
    if (!is.null(cl)) {
        cat(glue::glue("--- Stopping parallel backend early ---\n"))
        stopCluster(cl)
        registerDoSEQ()
    }
    quit(status = 1) # Exit script with non-zero status
}

# 7. Select Best Hyperparameters
cat(glue::glue("--- Selecting Best Hyperparameters ({TUNING_METRIC}) ---\n"))
best_params <- select_best_hyperparameters(tuning_results, TUNING_METRIC)
print(best_params)
# --- Save Best Parameters ---
best_params_rds_path <- file.path(MODEL_RESULTS_DIR, "best_params.rds")
cat(glue::glue("--- Saving Best Parameters to: {best_params_rds_path} ---\n"))
saveRDS(best_params, best_params_rds_path)

# 8. Finalize Workflow with Best Parameters
cat(glue::glue("--- Finalizing Workflow ---\n"))
final_workflow <- finalize_workflow(workflow_obj, best_params)
# --- Save Final Workflow ---
final_workflow_rds_path <- file.path(MODEL_RESULTS_DIR, "final_workflow.rds")
cat(glue::glue("--- Saving Final Workflow to: {final_workflow_rds_path} ---\n"))
saveRDS(final_workflow, final_workflow_rds_path)

# 9. Train Final Model on Full Training Data
cat(glue::glue("--- Training Final Model ({MODEL_NAME}) ---\n"))
final_model_fit <- tryCatch(
    {
        fit(final_workflow, data = train_data)
    },
    error = function(e) {
        warning(glue::glue("Final model training failed for model {MODEL_NAME}: {e$message}"), call. = FALSE)
        NULL
    }
)

# --- Save Final Model Fit ---
if (!is.null(final_model_fit)) {
    final_model_rds_path <- file.path(MODEL_RESULTS_DIR, "final_model_fit.rds")
    cat(glue::glue("--- Saving Final Model Fit to: {final_model_rds_path} ---\n"))
    saveRDS(final_model_fit, final_model_rds_path)
    cat("--- Final Model Trained ---\n")
} else {
    cat("--- Skipping model evaluation due to training failure ---\n")
    if (!is.null(cl)) {
        stopCluster(cl)
        registerDoSEQ()
    }
    quit(status = 1)
}

# 10. Make Predictions on Test Set
cat(glue::glue("--- Making Predictions on Test Data ---\n"))
test_predictions <- make_predictions(final_model_fit, test_data)

# 11. Evaluate Final Model on Test Set
cat(glue::glue("--- Evaluating Final Model on Test Data ---\n"))
test_metrics <- evaluate_model(
    predictions = test_predictions,
    actuals_data = test_data,
    truth_col = TARGET_VARIABLE,
    metrics_set = MODEL_METRICS
)
cat(glue::glue("--- Test Set Metrics: ---\n"))
print(test_metrics)

# --- Save Test Metrics ---
test_metrics_rds_path <- file.path(MODEL_RESULTS_DIR, "test_metrics.rds")
cat(glue::glue("--- Saving Test Metrics to: {test_metrics_rds_path} ---\n"))
saveRDS(test_metrics, test_metrics_rds_path)


# --- Stop Parallel Backend ---
if (!is.null(cl)) {
    cat(glue::glue("\n--- Stopping parallel backend ---\n"))
    stopCluster(cl)
    registerDoSEQ() # Register sequential backend
}
cat(glue::glue("\n=== Finished single model pipeline for {MODEL_NAME} ({TARGET_VARIABLE}) ===\n"))
