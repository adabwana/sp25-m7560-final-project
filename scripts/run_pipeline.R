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

# --- Parallel Processing Setup ---
suppressPackageStartupMessages(library(doParallel))
# --- Reduce cores for memory --- 
num_cores <- parallel::detectCores(logical = FALSE) # Get physical cores
num_cores_to_use <- 6 # Use a fixed, smaller number of cores
# num_cores <- parallel::detectCores(logical = FALSE) # Get physical cores
cat(glue::glue("\n--- Registering parallel backend with {num_cores_to_use} cores ---\n"))
cl <- makePSOCKcluster(num_cores_to_use)
registerDoParallel(cl)

# --- Source Helper Functions ---
# Construct full paths using here() for robustness
source(here::here("src/r/utils/data_utils.R"))
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))
source(here::here("src/r/tuning/tuning.R"))
source(here::here("src/r/training/training.R"))
source(here::here("src/r/evaluation/evaluation.R"))

# --- Configuration ---
# TODO: Move these to a config file/package (e.g., config::get())
DATA_FILENAME <- "train_engineered.csv" # Use only the training data
TEST_SPLIT_PROP <- 0.8 # Proportion of data for training set
TARGET_VARIABLE <- "Duration_In_Min" # Or "Occupancy"
CV_FOLDS <- 5 # Number of cross-validation folds
SEED <- 3 # For reproducibility
TUNING_METRIC <- "rmse" # Metric to select best hyperparameters (use "accuracy" or "roc_auc" for classification)
# Features to drop (matching Python preprocess.py, excluding targets and date columns handled in recipe)
FEATURES_TO_DROP <- c(
    "Student_IDs", "Semester", "Class_Standing", "Major", "Expected_Graduation",
    "Course_Name", "Course_Number", "Course_Type", "Course_Code_by_Thousands",
    "Check_Out_Time", "Session_Length_Category"
) # Add others as needed

# Define metric set based on task (regression in this case)
# TODO: Handle classification metrics if TARGET_VARIABLE changes
MODEL_METRICS <- metric_set(rmse, rsq, mae)


# --- Pipeline Execution ---

set.seed(SEED)

# 1. Load Data
cat(glue::glue("--- Loading Data ({DATA_FILENAME}) ---\n\n"))
full_data <- load_data(DATA_FILENAME)
cat(glue::glue("Full data rows: {nrow(full_data)}\n\n"))

# 2. Split Data into Training and Testing
cat(glue::glue("--- Splitting Data (Train: {TEST_SPLIT_PROP*100}%, Test: {(1-TEST_SPLIT_PROP)*100}%) ---\n\n"))
data_split <- rsample::initial_split(full_data, prop = TEST_SPLIT_PROP, strata = NULL) # Add strata if needed
train_data <- rsample::training(data_split)
test_data <- rsample::testing(data_split)
cat(glue::glue("Training data rows: {nrow(train_data)}, Test data rows: {nrow(test_data)}\n\n"))

# 3. Create Recipe
cat(glue::glue("--- Creating Recipe for Target: {TARGET_VARIABLE} ---\n\n"))
# NOTE: FEATURES_TO_DROP list is now correctly defined above
recipe_obj <- create_recipe(train_data, TARGET_VARIABLE, FEATURES_TO_DROP)
print(recipe_obj) # Print summary of recipe steps

# 4. Setup Resampling (Cross-Validation) for Tuning
cat(glue::glue("--- Setting up {CV_FOLDS}-fold Cross-Validation on Training Data ---\n\n"))
cv_folds <- rsample::vfold_cv(train_data, v = CV_FOLDS, strata = NULL) # Add strata = TARGET_VARIABLE if useful
print(cv_folds)

# --- Initialize Result Storage ---
all_tuning_results <- list()
all_best_params <- list()
all_final_models <- list()
all_test_metrics <- list()

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
cat(glue::glue("Models to process: {paste(model_names, collapse=", ")}\n\n"))

# --- Model Processing Loop ---
for (model_name in model_names) {
    cat(glue::glue("\n\n=== Processing Model: {model_name} ===\n\n"))
    
    # --- Force Garbage Collection --- 
    cat("(--- Running garbage collection ---)\n\n")
    gc()
    # -----------------------------

    # Get model spec and grid
    model_info <- model_list_to_use[[model_name]]
    # Basic check, should not happen if names() is used correctly
    # if (is.null(model_info)) {
    #   warning(glue::glue("Skipping model '{model_name}': Not found in the selected model list."))
    #   next
    # }
    model_spec <- model_info$spec
    model_grid <- model_info$grid
    cat(glue::glue("Model Spec: {class(model_spec)[1]}, Grid Size: {nrow(model_grid)}\n\n"))

    # 5. Build Workflow
    cat(glue::glue("--- Building Workflow ---\n\n"))
    workflow_obj <- build_workflow(recipe_obj, model_spec)
    # print(workflow_obj) # Keep output concise

    # 6. Tune Hyperparameters
    cat(glue::glue("--- Tuning Hyperparameters ---\n\n"))
    tune_control <- control_grid(verbose = FALSE, save_pred = FALSE, parallel_over = "everything")

    # Handle potential errors during tuning for a specific model
    tuning_results <- tryCatch({
        tune_model_grid(
            workflow = workflow_obj,
            resamples = cv_folds,
            grid = model_grid,
            metrics = MODEL_METRICS,
            control = tune_control
        )
    }, error = function(e) {
        warning(glue::glue("Tuning failed for model {model_name}: {e$message}"), call. = FALSE)
        NULL # Return NULL if tuning fails
    })
    
    # Store results and skip model if tuning failed
    all_tuning_results[[model_name]] <- tuning_results
    if (is.null(tuning_results)) {
        cat("--- Skipping model due to tuning failure ---\n")
        next # Skip to the next model
    }

    cat(glue::glue("--- Tuning Complete. Showing Best Metrics ({TUNING_METRIC}): ---\n\n"))
    print(show_best(tuning_results, metric = TUNING_METRIC, n = 5)) # Show only top 1

    # 7. Select Best Hyperparameters
    cat(glue::glue("--- Selecting Best Hyperparameters ---\n\n"))
    best_params <- select_best_hyperparameters(tuning_results, TUNING_METRIC)
    all_best_params[[model_name]] <- best_params
    print(best_params)

    # 8. Train Final Model
    cat(glue::glue("--- Training Final Model ---\n\n"))
    final_model <- tryCatch({
        train_final_model(
            workflow = workflow_obj,
            best_hyperparameters = best_params,
            training_data = train_data
        )
    }, error = function(e) {
        warning(glue::glue("Final model training failed for model {model_name}: {e$message}"), call. = FALSE)
        NULL
    })
    
    all_final_models[[model_name]] <- final_model
    if (is.null(final_model)) {
        cat("--- Skipping model evaluation due to training failure ---\n")
        next # Skip to the next model
    }
    cat("--- Final Model Trained ---\n\n")

    # 9. Make Predictions on Test Set
    cat(glue::glue("--- Making Predictions on Test Data ---\n\n"))
    test_predictions <- make_predictions(final_model, test_data)

    # 10. Evaluate Final Model on Test Set
    cat(glue::glue("--- Evaluating Final Model on Test Data ---\n\n"))
    test_metrics <- evaluate_model(
        predictions = test_predictions,
        actuals_data = test_data,
        truth_col = TARGET_VARIABLE,
        metrics_set = MODEL_METRICS
    )
    all_test_metrics[[model_name]] <- test_metrics
    cat(glue::glue("--- Test Set Metrics: ---\n\n"))
    print(test_metrics)

    cat(glue::glue("\n=== Finished Model: {model_name} ===\n\n"))
} # --- End Model Processing Loop ---

# --- Summarize Results ---
cat(glue::glue("\n\n=== Overall Test Set Performance Summary ===\n"))

best_model_name <- NULL # Initialize

if (length(all_test_metrics) > 0) {
    # Combine metrics into a single table
    summary_metrics <- bind_rows(all_test_metrics, .id = "model_name") %>%
        # Pivot wider for easier comparison
        tidyr::pivot_wider(names_from = .metric, values_from = .estimate) %>%
        # Sort by rmse
        arrange(rmse)
    
    print(summary_metrics)
    
    # Find best model based on the primary tuning metric
    best_model_summary <- summary_metrics %>%
        # Ensure the tuning metric column exists and is not NA
        filter(!is.na(.data[[TUNING_METRIC]])) 
    
    if(nrow(best_model_summary) > 0) {
        # Use slice_min for metrics where lower is better (rmse, mae)
        # Use slice_max for metrics where higher is better (rsq, accuracy, roc_auc)
        if (TUNING_METRIC %in% c("rmse", "mae")) {
            best_model_summary <- best_model_summary %>%
                slice_min(order_by = .data[[TUNING_METRIC]], n = 1, with_ties = FALSE)
        } else {
            best_model_summary <- best_model_summary %>%
                slice_max(order_by = .data[[TUNING_METRIC]], n = 1, with_ties = FALSE)
        }
        
        best_model_name <- best_model_summary %>%
            pull(model_name)
            
        cat(glue::glue("\nBest model based on test set {TUNING_METRIC}: {best_model_name}\n"))
        
        # --- Save the Best Model --- 
        if (!is.null(best_model_name) && best_model_name %in% names(all_final_models)) {
            best_model_object <- all_final_models[[best_model_name]]
            
            if (!is.null(best_model_object)) {
                save_dir <- here::here("artifacts", "r", "models")
                dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
                save_filename <- glue::glue("{TARGET_VARIABLE}_best_model_{best_model_name}.rds")
                save_path <- file.path(save_dir, save_filename)
                
                cat(glue::glue("--- Saving best model ({best_model_name}) for target '{TARGET_VARIABLE}' to {save_path} ---\n"))
                saveRDS(best_model_object, file = save_path)
                cat("--- Model saved successfully ---\n")
            } else {
                 warning(glue::glue("Best model object for '{best_model_name}' is NULL, cannot save."), call. = FALSE)
            }
        } else {
             warning(glue::glue("Best model name '{best_model_name}' not found in results or is NULL, cannot save."), call. = FALSE)
        }
        # --- End Save --- 
        
    } else {
         cat(glue::glue("\nCould not determine best model based on {TUNING_METRIC} (no valid results found).\n"))
    }
    
} else {
    cat("\nNo models completed successfully to summarize results.\n")
}

cat(glue::glue("--- R Pipeline Execution Complete ---\n\n"))

# --- Stop Parallel Backend ---
cat(glue::glue("--- Stopping parallel backend ---\n\n"))
stopCluster(cl)
registerDoSEQ() # Register sequential backend
