library(testthat)
library(recipes)
library(tibble)
library(lubridate)
library(here)

# Source the recipe definition file
# Use here() to ensure path is correct from project root
source(here::here("src/r/recipes/recipes.R"))

context("Recipe Creation and Preprocessing Steps")

# --- Remove Placeholder Function ---
# create_recipe <- function(data, outcome_var, cols_to_drop) { ... removed ... }
# --- End Remove Placeholder ---

test_that("create_recipe generates a valid recipe object", {
    # Create minimal sample data mimicking train_engineered.csv structure
    sample_data <- tibble::tribble(
        ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date, # Added date cols for removal step
        "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
        "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10",
        "S3", "10:30:00", "CompSci", 90, 1, 11.0, "Senior", "2024-01-12", "2024-01-08", "2024-05-10",
        "S4", "12:15:00", "Physics", 45, 1, 9.1, "Sophomore", "2024-01-13", "2024-01-08", "2026-05-10"
    )

    # Define parameters based on preprocess.py + actual targets
    target_duration <- "Duration_In_Min"
    features_to_drop_py <- c(
        "Student_IDs", "Semester", # Keep Class_Standing & Major for dummy test
        "Expected_Graduation",
        "Course_Name", "Course_Number", "Course_Type", "Course_Code_by_Thousands",
        "Check_Out_Time", "Session_Length_Category"
    ) # Exclude targets from this list

    # Create recipe using the actual function
    rec <- create_recipe(sample_data, target_duration, features_to_drop_py)

    # Basic check: Is it a recipe object?
    expect_s3_class(rec, "recipe")

    # Check if steps seem reasonable (count steps, check classes)
    expect_gt(length(rec$steps), 5) # Should have several steps

    # Remove the skip message
    # skip("Skipping test until create_recipe is implemented in src/r/recipes/recipes.R")
})

test_that("Prepared recipe has expected structure (Duration target)", {
    # Create minimal sample data
    sample_data <- tibble::tribble(
        ~Student_IDs, ~Check_In_Time, ~Major, ~Duration_In_Min, ~Occupancy, ~NumericFeature, ~Class_Standing, ~Check_In_Date, ~Semester_Date, ~Expected_Graduation_Date, # Added date cols for removal step
        "S1", "10:30:00", "CompSci", 120, 1, 10.5, "Senior", "2024-01-10", "2024-01-08", "2024-05-10",
        "S2", "11:00:00", "Math", 60, 1, 8.2, "Junior", "2024-01-11", "2024-01-08", "2025-05-10",
        "S3", "10:30:00", "CompSci", 90, 1, 11.0, "Senior", "2024-01-12", "2024-01-08", "2024-05-10",
        "S4", "12:15:00", "Physics", 45, 1, 9.1, "Sophomore", "2024-01-13", "2024-01-08", "2026-05-10"
    )

    target_duration <- "Duration_In_Min"
    # Note: Class_Standing and Major are *not* in this list, so they should be dummied
    features_to_drop_py <- c(
        "Student_IDs", "Semester",
        "Expected_Graduation", # This one is in the data, should be dropped
        "Course_Name", "Course_Number", "Course_Type", "Course_Code_by_Thousands", # Not in sample data, handled by any_of
        "Check_Out_Time", "Session_Length_Category"
    ) # Not in sample data, handled by any_of

    # Create recipe using the actual function
    rec <- create_recipe(sample_data, target_duration, features_to_drop_py)

    # Prep and bake
    prepared_rec <- prep(rec, training = sample_data)
    baked_data <- bake(prepared_rec, new_data = NULL) # Bake training data

    # Expected columns after processing (Order might vary slightly):
    # - Duration_In_Min (outcome)
    # - NumericFeature (numeric predictor, normalized)
    # - Check_In_Time_Minutes (numeric predictor, normalized)
    # - Major_Math, Major_Physics (dummies from Major, CompSci is baseline, numeric, normalized)
    # - Class_Standing_Senior, Class_Standing_Sophomore (dummies from Class_Standing, Junior is baseline, numeric, normalized)
    # Dropped: Student_IDs, Check_In_Time, Major, Occupancy, Class_Standing,
    #          Check_In_Date, Semester_Date, Expected_Graduation_Date
    # ZV/Normalization added
    expected_cols <- c(
        "Duration_In_Min", "NumericFeature", "Check_In_Time_Minutes",
        "Major_Math", "Major_Physics", "Class_Standing_Senior", "Class_Standing_Sophomore"
    )

    # Check column names
    # Use intersect to avoid issues with potential order differences or extra meta-cols
    expect_true(all(expected_cols %in% colnames(baked_data)))
    expect_true(all(colnames(baked_data) %in% c(expected_cols, target_duration))) # Allow only expected + outcome
    # More precise check:
    expect_equal(sort(setdiff(colnames(baked_data), target_duration)), sort(setdiff(expected_cols, target_duration)))

    # Check data types (after normalization, all predictors should be numeric)
    expect_type(baked_data$Duration_In_Min, "double") # Outcome type might depend on original
    predictor_cols <- setdiff(colnames(baked_data), target_duration)
    for (col in predictor_cols) {
        expect_type(baked_data[[col]], "double")
    }

    # Check normalization (mean should be ~0, sd ~1 for predictors)
    for (col in predictor_cols) {
        expect_equal(mean(baked_data[[col]]), 0, tolerance = 1e-6)
        expect_equal(sd(baked_data[[col]]), 1, tolerance = 1e-6)
    }

    # Remove the skip message
    # skip("Skipping test until create_recipe is implemented and uses real data logic.")
})

# Add similar test_that block for 'Occupancy' target if needed
