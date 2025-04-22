#!/usr/bin/env Rscript

# --- Load Libraries ---
# Using suppressPackageStartupMessages to minimize console noise
suppressPackageStartupMessages(library(here)) # For locating files relative to project root
suppressPackageStartupMessages(library(dplyr)) # For data manipulation
suppressPackageStartupMessages(library(rsample)) # For data splitting and CV folds
suppressPackageStartupMessages(library(tune)) # For hyperparameter tuning
suppressPackageStartupMessages(library(yardstick)) # For model evaluation metrics
suppressPackageStartupMessages(library(workflows)) # For combining recipes and models
suppressPackageStartupMessages(library(parsnip)) # For model specifications
suppressPackageStartupMessages(library(recipes)) # For preprocessing
suppressPackageStartupMessages(library(readr)) # For reading/writing data
suppressPackageStartupMessages(library(glue)) # For formatted string output
suppressPackageStartupMessages(library(config)) # For configuration management
suppressPackageStartupMessages(library(tidyr)) # For pivot_wider
suppressPackageStartupMessages(library(stringr)) # For string manipulation

# --- Source Helper Functions ---
source(here::here("src/r/utils/config_utils.R"))
source(here::here("src/r/utils/data_utils.R"))
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))
source(here::here("src/r/tuning/tuning.R"))
source(here::here("src/r/training/training.R"))
source(here::here("src/r/evaluation/evaluation.R"))

# --- Load Configuration ---
cfg <- load_config()

# --- Inspect Loaded Config ---
cat("\n--- Structure of Loaded Config (cfg) ---\n")
str(cfg)
cat("---------------------------------------\n\n")

# --- Configuration ---
# Access values directly from the merged config object
DATA_FILENAME <- get_config_value(cfg, "data.filename")
# Define split proportions explicitly
INITIAL_SPLIT_PROP <- get_config_value(cfg, "data.trainval_holdout_prop") # Load from config
# Note: TEST_SPLIT_PROP from config is now INITIAL_SPLIT_PROP. Holdout is 1 - INITIAL_SPLIT_PROP
TARGET_VARIABLE <- cfg$data$target_variable
CV_FOLDS <- cfg$model$cv_folds
SEED <- cfg$model$seed
TUNING_METRIC <- cfg$model$tuning_metric
FEATURES_TO_DROP <- cfg$data$features_to_drop
NUM_TOP_MODELS <- 5 # Number of top models to evaluate on holdout

# --- Check if TARGET_VARIABLE is loaded ---
if (is.null(TARGET_VARIABLE) || length(TARGET_VARIABLE) == 0) {
    stop("TARGET_VARIABLE was not loaded correctly from the configuration.")
}

# Add the *other* target variable to the drop list
if (TARGET_VARIABLE == "Duration_In_Min") {
    FEATURES_TO_DROP <- c(FEATURES_TO_DROP, "Occupancy")
} else if (TARGET_VARIABLE == "Occupancy") {
    FEATURES_TO_DROP <- c(FEATURES_TO_DROP, "Duration_In_Min")
} else {
    warning(glue::glue("Unknown TARGET_VARIABLE: '{TARGET_VARIABLE}'. Not adding default other target to FEATURES_TO_DROP."))
}

# Define metric set based on task
MODEL_METRICS <- metric_set(rmse, rsq, mae)

# --- Parallel Processing Setup ---
suppressPackageStartupMessages(library(doParallel))
if (get_config_value(cfg, "parallel.enabled", TRUE)) {
    num_cores_to_use <- get_config_value(cfg, "parallel.num_cores")
    cat(glue::glue("\n--- Registering parallel backend with {num_cores_to_use} cores ---\n"))
    cl <- makePSOCKcluster(num_cores_to_use)
    registerDoParallel(cl)
}

# --- Pipeline Execution ---

set.seed(SEED)

# 1. Load Data
cat(glue::glue("--- Loading Data ({DATA_FILENAME}) ---\n\n"))
full_data <- load_data(DATA_FILENAME)
cat(glue::glue("Full data rows: {nrow(full_data)}\n\n"))

# 2. Split Data into Train/Validation and Holdout (Test)
holdout_prop <- 1 - INITIAL_SPLIT_PROP
cat(glue::glue("--- Splitting Data (Train/Val: {INITIAL_SPLIT_PROP*100}%, Holdout: {holdout_prop*100}%) ---\n\n"))
# This is the crucial split for holdout evaluation
train_val_split <- rsample::initial_split(full_data, prop = INITIAL_SPLIT_PROP, strata = NULL) # Add strata if needed
train_val_data <- rsample::training(train_val_split)
holdout_data <- rsample::testing(train_val_split)
cat(glue::glue("Train/Val data rows: {nrow(train_val_data)}, Holdout data rows: {nrow(holdout_data)}\n\n"))

# 3. Create Recipe (using only Train/Val data for setup)
cat(glue::glue("--- Creating Recipe for Target: {TARGET_VARIABLE} (using Train/Val data) ---\n\n"))
recipe_obj <- create_recipe(train_val_data, TARGET_VARIABLE, FEATURES_TO_DROP)
print(recipe_obj)

# 4. Setup Resampling (Cross-Validation) for Tuning (using Train/Val data)
cat(glue::glue("--- Setting up {CV_FOLDS}-fold Cross-Validation on Train/Val Data ---\n\n"))
cv_folds <- rsample::vfold_cv(train_val_data, v = CV_FOLDS, strata = NULL) # Add strata = TARGET_VARIABLE if useful
# print(cv_folds) # Keep output concise

# --- Initialize Result Storage ---
tuning_results_list <- list()
best_params_list <- list()
best_cv_metrics_list <- list() # To store the best CV metric for each model

# --- Determine Model List ---
model_list_to_use <- if (TARGET_VARIABLE == "Duration_In_Min") {
    cat(glue::glue("--- Using model list: model_list_duration ---\n\n"))
    model_list_duration
} else if (TARGET_VARIABLE == "Occupancy") {
    cat(glue::glue("--- Using model list: model_list_occupancy ---\n\n"))
    model_list_occupancy
} else {
    stop(glue::glue("Unknown TARGET_VARIABLE: {TARGET_VARIABLE}"))
}

model_names <- names(model_list_to_use)
cat(glue::glue("Models to process: {paste(model_names, collapse=\", \")}\n\n"))

# --- Model Tuning Loop ---
cat(glue::glue("

=== Starting Model Tuning Phase ===

"))
for (model_name in model_names) {
    cat(glue::glue("
--- Tuning Model: {model_name} ---
"))

    # --- Force Garbage Collection ---
    cat("(--- Running garbage collection ---)
")
    gc()
    # -----------------------------

    model_info <- model_list_to_use[[model_name]]
    model_spec <- model_info$spec
    model_grid <- model_info$grid
    # cat(glue::glue("Model Spec: {class(model_spec)[1]}, Grid Size: {nrow(model_grid)}\n")) # Concise output

    # 5. Build Workflow
    # cat(glue::glue("--- Building Workflow ---
    # ")) # Concise output
    workflow_obj <- build_workflow(recipe_obj, model_spec)

    # 6. Tune Hyperparameters using CV folds from Train/Val data
    # cat(glue::glue("--- Tuning Hyperparameters ---
    # ")) # Concise output
    tune_control <- control_grid(verbose = FALSE, save_pred = FALSE, parallel_over = "everything")

    tuning_results <- tryCatch(
        {
            tune_model_grid(
                workflow = workflow_obj,
                resamples = cv_folds, # Use CV folds from Train/Val data
                grid = model_grid,
                metrics = MODEL_METRICS,
                control = tune_control
            )
        },
        error = function(e) {
            warning(glue::glue("Tuning failed for model {model_name}: {e$message}"), call. = FALSE)
            NULL
        }
    )

    tuning_results_list[[model_name]] <- tuning_results
    if (is.null(tuning_results)) {
        cat("--- Skipping model due to tuning failure ---
")
        next
    }

    cat(glue::glue("--- Tuning Complete. Best CV {TUNING_METRIC}: ---
"))
    best_cv_result <- show_best(tuning_results, metric = TUNING_METRIC, n = 5)
    print(best_cv_result)

    # 7. Select Best Hyperparameters based on CV
    # cat(glue::glue("--- Selecting Best Hyperparameters ---
    # ")) # Concise output
    best_params <- select_best_hyperparameters(tuning_results, TUNING_METRIC)
    best_params_list[[model_name]] <- best_params
    # print(best_params) # Concise output

    # Store the best CV metric value for ranking later
    best_cv_metrics_list[[model_name]] <- best_cv_result %>% pull(mean)

    cat(glue::glue("--- Finished Tuning: {model_name} ---
"))
} # --- End Model Tuning Loop ---

# --- Select Top Models Based on CV Performance ---
cat(glue::glue("

=== Selecting Top {NUM_TOP_MODELS} Models based on CV Performance ({TUNING_METRIC}) ===

"))

# Convert the list of best CV metrics to a data frame
cv_performance_summary <- tibble::enframe(best_cv_metrics_list, name = "model_name", value = "best_cv_metric") %>%
    filter(!is.na(best_cv_metric)) # Remove models that failed tuning

if (nrow(cv_performance_summary) == 0) {
    stop("No models completed tuning successfully. Cannot proceed.")
}

# Determine sort order (lower is better for rmse/mae, higher otherwise)
sort_descending <- !(TUNING_METRIC %in% c("rmse", "mae"))

# Sort models
ranked_models <- cv_performance_summary %>%
    arrange(if (sort_descending) desc(best_cv_metric) else best_cv_metric)

top_models_ranked <- ranked_models %>%
    slice_head(n = NUM_TOP_MODELS)

cat("--- Top Models based on CV Performance: ---
")
print(top_models_ranked)

# --- Evaluate Top Models on Holdout Set ---
cat(glue::glue("

=== Evaluating Top {nrow(top_models_ranked)} Models on Holdout Set ===

"))

all_holdout_results <- list()
all_holdout_metrics <- list()

for (model_name in top_models_ranked$model_name) {
    cat(glue::glue("
--- Evaluating Model on Holdout: {model_name} ---
"))

    # Get the original model spec and best parameters for this model
    model_info <- model_list_to_use[[model_name]]
    model_spec <- model_info$spec
    best_params <- best_params_list[[model_name]]

    # Create the workflow and finalize it
    workflow_obj <- build_workflow(recipe_obj, model_spec)
    final_workflow <- finalize_workflow(workflow_obj, best_params)

    # Use last_fit() - trains on train_val_data, evaluates on holdout_data
    cat("--- Running last_fit() ---
")
    holdout_fit <- tryCatch(
        {
            last_fit(
                final_workflow,
                split = train_val_split # Use the original 75/25 split object
            )
        },
        error = function(e) {
            warning(glue::glue("last_fit() failed for model {model_name}: {e$message}"), call. = FALSE)
            NULL
        }
    )

    all_holdout_results[[model_name]] <- holdout_fit

    if (is.null(holdout_fit)) {
        cat("--- Skipping model holdout evaluation due to last_fit() failure ---
")
        next
    }

    # Collect metrics from the holdout evaluation
    holdout_metrics <- collect_metrics(holdout_fit)
    all_holdout_metrics[[model_name]] <- holdout_metrics

    cat("--- Holdout Set Metrics: ---
")
    print(holdout_metrics)

    cat(glue::glue("--- Finished Holdout Evaluation: {model_name} ---
"))
} # --- End Holdout Evaluation Loop ---


# --- Summarize Holdout Results ---
cat(glue::glue("

=== Overall Holdout Set Performance Summary (Top {nrow(top_models_ranked)} Models) ===
"))

best_holdout_model_name <- NULL # Initialize
best_holdout_model_object <- NULL # Initialize

if (length(all_holdout_metrics) > 0) {
    # Combine metrics into a single table
    summary_holdout_metrics <- bind_rows(all_holdout_metrics, .id = "model_name") %>%
        # Pivot wider for easier comparison
        tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

    # Merge with CV performance for context (optional but helpful)
    summary_holdout_metrics <- left_join(summary_holdout_metrics, ranked_models, by = "model_name") %>%
        select(model_name, best_cv_metric, everything()) # Reorder cols

    # Arrange by the tuning metric on the holdout set
    if (TUNING_METRIC %in% names(summary_holdout_metrics)) {
        summary_holdout_metrics <- summary_holdout_metrics %>%
            filter(!is.na(.data[[TUNING_METRIC]])) %>%
            arrange(if (sort_descending) desc(.data[[TUNING_METRIC]]) else .data[[TUNING_METRIC]])
    } else {
        warning(glue::glue("Tuning metric '{TUNING_METRIC}' not found in holdout metrics. Cannot sort summary table by it."))
        # Arrange by model name as fallback
        summary_holdout_metrics <- summary_holdout_metrics %>% arrange(model_name)
    }


    print(summary_holdout_metrics)

    # Find best model based on the primary tuning metric ON THE HOLDOUT SET
    if (nrow(summary_holdout_metrics) > 0) {
        best_holdout_model_summary <- summary_holdout_metrics %>%
            slice(1) # Already sorted, take the first row

        best_holdout_model_name <- best_holdout_model_summary %>% pull(model_name)
        best_holdout_metric_value <- best_holdout_model_summary %>% pull(.data[[TUNING_METRIC]])

        cat(glue::glue("
Best model based on holdout set {TUNING_METRIC}: {best_holdout_model_name} ({TUNING_METRIC} = {round(best_holdout_metric_value, 4)})\n"))

        # --- Save the Best Model (trained on Train/Val) ---
        if (!is.null(best_holdout_model_name) && best_holdout_model_name %in% names(all_holdout_results)) {
            # Extract the workflow fitted on the train_val_data from the last_fit object
            best_holdout_fit_result <- all_holdout_results[[best_holdout_model_name]]

            if (!is.null(best_holdout_fit_result)) {
                best_holdout_model_object <- extract_workflow(best_holdout_fit_result)

                # Format the metric value for filename
                formatted_metric_value <- format(round(best_holdout_metric_value, 3), nsmall = 3) %>%
                    stringr::str_replace_all("\\.", "-") # Replace . with -

                save_dir <- here::here("artifacts", "r", "models")
                dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
                # Construct filename including formatted metric value and indicating holdout selection
                save_filename <- glue::glue("{TARGET_VARIABLE}_best_holdout_{TUNING_METRIC}_{formatted_metric_value}_{best_holdout_model_name}.rds")
                save_path <- file.path(save_dir, save_filename)

                cat(glue::glue("--- Saving best model's workflow ({best_holdout_model_name}), trained on Train/Val data, to {save_path} ---
"))
                saveRDS(best_holdout_model_object, file = save_path)
                cat("--- Model workflow saved successfully ---
")
            } else {
                warning(glue::glue("last_fit result for the best model '{best_holdout_model_name}' is NULL, cannot save workflow."), call. = FALSE)
            }
        } else {
            warning(glue::glue("Best holdout model name '{best_holdout_model_name}' not found in holdout results or is NULL, cannot save."), call. = FALSE)
        }
        # --- End Save ---
    } else {
        cat(glue::glue("
Could not determine best model based on holdout set {TUNING_METRIC} (no valid holdout results found).
"))
    }
} else {
    cat("
No models completed holdout evaluation successfully to summarize results.
")
}

cat(glue::glue("
--- R Pipeline Execution Complete ---

"))

# --- Stop Parallel Backend ---
if (exists("cl") && !is.null(cl)) {
    cat(glue::glue("--- Stopping parallel backend ---

"))
    stopCluster(cl)
    registerDoSEQ() # Register sequential backend
}
