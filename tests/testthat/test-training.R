library(testthat)
library(workflows)
library(recipes)
library(parsnip)
library(tune)
library(tibble)
library(dplyr)
library(here)

# Source dependencies
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
source(here::here("src/r/workflows/workflows.R"))
source(here::here("src/r/tuning/tuning.R")) # Needed for test setup
# Source the training function
source(here::here("src/r/training/training.R"))

context("Final Model Training")

# --- Remove Placeholder Function ---
# train_final_model <- function(...) { ... removed ...}
# --- End Placeholder Function ---


# --- Test Setup ---
# Reuse setup from tuning tests, but maybe slightly larger data for fitting simulation
sample_data_train_small <- tibble::tribble(
    ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date,
    "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
    "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10",
    "S3", "10:45:00", "CompSci", 95, 1, 11.1, "Senior", "2024-01-12", "2024-01-08", "2024-05-10",
    "S4", "12:15:00", "Physics", 45, 1, 9.1, "Sophomore", "2024-01-13", "2024-01-08", "2026-05-10",
    "S5", "09:00:00", "Math", 70, 1, 8.5, "Junior", "2024-01-14", "2024-01-08", "2025-05-10"
)
sample_data_train <- bind_rows(replicate(6, sample_data_train_small, simplify = FALSE)) %>%
    mutate(Duration_In_Min = Duration_In_Min + rnorm(n(), 0, 5)) %>%
    mutate(Student_IDs = paste0(Student_IDs, "_", row_number()))

# Create workflow
target_duration <- "Duration_In_Min"
features_to_drop_py <- c("Student_IDs", "Semester", "Expected_Graduation", "Check_Out_Time", "Session_Length_Category")
train_recipe <- create_recipe(sample_data_train, target_duration, features_to_drop_py)
train_spec <- mars_spec # Using MARS spec
train_wf <- build_workflow(train_recipe, train_spec)

# Get sample "best" hyperparameters (simulate result of tuning)
best_params_sample <- head(mars_grid, 1)


# --- Tests ---
test_that("train_final_model returns a fitted workflow object", {
    # Use actual function
    fitted_model <- train_final_model(train_wf, best_params_sample, sample_data_train)

    # Check if it's a workflow object
    expect_s3_class(fitted_model, "workflow")
    # Check if it's marked as trained
    expect_true(fitted_model$trained)
})

test_that("Fitted workflow contains a trained recipe", {
    # Fit the model using the function being tested
    fitted_model <- train_final_model(train_wf, best_params_sample, sample_data_train)

    # Check that the recipe inside the *fitted* workflow is prepped/trained
    # Use extract_recipe with estimated = TRUE. If it runs without error, it implies the recipe was trained.
    expect_no_error(workflows::extract_recipe(fitted_model, estimated = TRUE))
})

# --- NEW TEST ---
test_that("Fitted workflow from train_final_model can make predictions", {
    # Fit the model
    fitted_model <- train_final_model(train_wf, best_params_sample, sample_data_train)

    # Attempt to predict on the training data (or a subset)
    predictions <- NULL
    expect_no_error(predictions <- predict(fitted_model, new_data = head(sample_data_train, 5)))

    # Check predictions format
    expect_s3_class(predictions, "tbl_df")
    expect_true(".pred" %in% names(predictions))
    expect_equal(nrow(predictions), 5)
})
# --- END NEW TEST ---

test_that("train_final_model handles invalid inputs", {
    # Use actual function
    expect_error(train_final_model("not a workflow", best_params_sample, sample_data_train), "`workflow` must be a workflow object")
    expect_error(train_final_model(train_wf, mars_grid, sample_data_train), "`best_hyperparameters` must be.*with exactly one row") # Use grid with > 1 row
    expect_error(train_final_model(train_wf, best_params_sample, "not data"), "`training_data` must be a data frame")
})
