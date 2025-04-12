library(testthat)
library(workflows)
library(recipes)
library(parsnip)
library(tibble)
library(here)

# Source dependencies: recipe creation function and model specs
source(here::here("src/r/recipes/recipes.R"))
source(here::here("src/r/models/models.R"))
# Source the workflow function
source(here::here("src/r/workflows/workflows.R"))

context("Workflow Construction")

# --- Remove Placeholder Function ---
# build_workflow <- function(recipe, model_spec) { ... removed ... }
# --- End Remove Placeholder ---

# --- Test Setup ---
# Create minimal sample data for recipe creation
sample_data_wf <- tibble::tribble(
    ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date,
    "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
    "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10"
)

# Create a sample recipe (using the function from recipes.R)
target_duration <- "Duration_In_Min"
features_to_drop_py <- c("Student_IDs", "Semester", "Expected_Graduation", "Check_Out_Time", "Session_Length_Category")
sample_recipe <- create_recipe(sample_data_wf, target_duration, features_to_drop_py)

# Use one of the defined model specs (from models.R)
sample_spec <- mars_spec

# --- Tests ---
test_that("build_workflow creates a valid workflow object", {
    # Create workflow using the actual function
    wf <- build_workflow(sample_recipe, sample_spec)

    # Check class
    expect_s3_class(wf, "workflow")

    # Check for essential components
    expect_true(!is.null(wf$pre)) # Preprocessor (recipe)
    expect_true(!is.null(wf$fit)) # Fitter (model spec)
})

test_that("Workflow contains the correct recipe and model spec", {
    # Create workflow using the actual function
    wf <- build_workflow(sample_recipe, sample_spec)

    # Extract components
    # Using the explicit extractor functions is more robust
    wf_recipe <- workflows::extract_recipe(wf, estimated = FALSE)
    wf_spec <- workflows::extract_spec_parsnip(wf)

    # Compare with inputs
    # Comparing recipes directly can be tricky due to environments.
    # Comparing the steps list is usually sufficient.
    expect_equal(wf_recipe$steps, sample_recipe$steps)
    expect_equal(wf_spec, sample_spec)
})

test_that("build_workflow handles invalid input types", {
    # Test with incorrect inputs using the actual function
    expect_error(build_workflow("not a recipe", sample_spec), "`recipe` must be a recipe object")
    expect_error(build_workflow(sample_recipe, "not a model_spec"), "`model_spec` must be a model_spec object")
})
