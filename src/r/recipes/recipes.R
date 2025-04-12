# Load necessary libraries
library(recipes)
library(lubridate) # For hms, hour, minute
library(dplyr) # For any_of selector
library(rlang) # For sym function

#' Create Preprocessing Recipe
#'
#' Defines the preprocessing steps based on the tidymodels framework,
#' mimicking the logic from the Python preprocess.py script.
#'
#' @param data The *training* data frame/tibble used to define the recipe steps.
#' @param outcome_var A string specifying the name of the outcome variable
#'   (e.g., "Duration_In_Min" or "Occupancy").
#' @param features_to_drop A character vector of feature names to explicitly drop
#'   *before* other preprocessing steps. This list should NOT include the target variables.
#'
#' @return A recipe object with defined preprocessing steps.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming train_data is loaded
#' features_to_drop_list <- c(
#'     "Student_IDs", "Semester", "Class_Standing", "Major",
#'     "Expected_Graduation", "Course_Name", "Course_Number",
#'     "Course_Type", "Course_Code_by_Thousands",
#'     "Check_Out_Time", "Session_Length_Category"
#' )
#'
#' duration_recipe <- create_recipe(train_data, "Duration_In_Min", features_to_drop_list)
#' occupancy_recipe <- create_recipe(train_data, "Occupancy", features_to_drop_list)
#'
#' # Print the recipe
#' # print(duration_recipe)
#'
#' # Prep the recipe
#' # prepped_recipe <- prep(duration_recipe, training = train_data)
#' # print(prepped_recipe)
#'
#' # Bake the data
#' # baked_train_data <- bake(prepped_recipe, new_data = NULL)
#' # baked_test_data <- bake(prepped_recipe, new_data = test_data) # Assuming test_data exists
#' }
create_recipe <- function(data, outcome_var, features_to_drop) {
    # Ensure outcome_var is valid
    stopifnot(outcome_var %in% c("Duration_In_Min", "Occupancy"))
    stopifnot(is.character(features_to_drop))
    stopifnot(is.data.frame(data))

    # Determine the *other* target variable to drop
    other_target <- setdiff(c("Duration_In_Min", "Occupancy"), outcome_var)

    # Combine explicit drops and the other target
    all_drops <- unique(c(features_to_drop, other_target))

    # Convert outcome_var string to symbol for formula
    outcome_sym <- rlang::sym(outcome_var)

    # --- Define Recipe ---
    rec <- recipe(data) %>% # Start with data, formula added later
        # 1. Set the outcome variable role
        update_role(!!outcome_sym, new_role = "outcome") %>%
        # Identify remaining predictors (update roles for ID-like/date columns first)
        # Note: This assumes 'features_to_drop' contains most ID/unwanted cols.
        # We might need to explicitly assign roles if columns are kept initially
        # for transformations (like Check_In_Time before conversion).
        update_role(all_nominal(), -all_outcomes(), new_role = "predictor") %>%
        update_role(all_numeric(), -all_outcomes(), new_role = "predictor") %>%
        # Add specific roles for columns if needed before dropping
        # update_role(Check_In_Time, new_role = "time_col_to_process") %>%

        # 2. Drop specified columns (robustly handles non-existent columns)
        # We use step_rm which needs explicit variable names. Selector any_of helps.
        step_rm(dplyr::any_of(all_drops)) %>%
        # 3. Convert Check_In_Time to minutes since midnight
        # This assumes Check_In_Time exists at this point and is 'HH:MM:SS'
        # It operates on the data subset currently being processed by the recipe step.
        step_mutate(Check_In_Time_Minutes = (lubridate::hour(lubridate::hms(Check_In_Time)) * 60 +
            lubridate::minute(lubridate::hms(Check_In_Time)))) %>%
        # Remove original Check_In_Time column after processing
        step_rm(Check_In_Time) %>%
        # Remove other date columns identified in Python preprocess.py
        step_rm(dplyr::any_of(c("Check_In_Date", "Semester_Date", "Expected_Graduation_Date"))) %>%
        # 4. Impute missing numeric data (example: mean imputation)
        # Add imputation if required by models (e.g., MARS can handle some NA)
        step_impute_mean(all_numeric_predictors()) %>%
        # 5. Handle unseen factor levels for nominal predictors before dummy coding
        step_novel(all_nominal_predictors()) %>%
        # 6. Create dummy variables for nominal predictors (excluding outcome)
        # one_hot = FALSE mimics pandas get_dummies(drop_first=True)
        step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
        # 7. Remove zero-variance predictors (often needed after dummy coding)
        step_zv(all_predictors()) %>%
        # 8. Normalize numeric predictors (optional, often good practice)
        step_normalize(all_numeric_predictors())

    # Return the recipe object
    return(rec)
}
