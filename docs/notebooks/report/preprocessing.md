# Preprocessing Pipelines

Consistent and appropriate preprocessing is crucial for effective modeling. While our feature engineering (detailed previously) was performed entirely in R, the final preprocessing steps before model training were handled slightly differently within the R and Python frameworks.

## R `tidymodels` Preprocessing (`recipes`)

For the models trained using the R `tidymodels` ecosystem (MARS, Random Forest, XGBoost), we defined a standardized preprocessing pipeline using the `recipes` package. This object, created by the `create_recipe` function in `src/r/recipes/recipes.R`, encapsulated the necessary steps to prepare the engineered data for these specific algorithms.

Key steps included:

- **Role Definition**: Assigning 'outcome' and 'predictor' roles to variables.
- **Variable Removal**: Removing identifier columns, raw date/time fields (after extracting relevant features like minutes past midnight), and the *other* target variable not being predicted in a given run.
- **Time Conversion**: Transforming `Check_In_Time` into a numeric `Check_In_Time_Minutes` feature.
- **Imputation**: Handling missing numeric values using mean imputation (`step_impute_mean`).
- **Categorical Handling**: Preparing for potential new factor levels during resampling (`step_novel`) and converting nominal predictors to numeric using dummy variables (`step_dummy` with `one_hot = FALSE` to mimic pandas `drop_first=True`).
- **Variance Filtering**: Removing predictors with zero variance (`step_zv`), often necessary after dummy coding.
- **Normalization**: Centering and scaling all numeric predictors (`step_normalize`).

This approach ensured that the data fed into the `tidymodels` workflows was consistently processed according to the needs of the algorithms.

```r
# Example recipe creation (from src/r/recipes/recipes.R)
create_recipe <- function(data, outcome_var, features_to_drop) {
    # ... (Input validation) ...
    other_target <- setdiff(c("Duration_In_Min", "Occupancy"), outcome_var)
    all_drops <- unique(c(features_to_drop, other_target))
    outcome_sym <- rlang::sym(outcome_var)

    rec <- recipe(data) %>%
        update_role(!!outcome_sym, new_role = "outcome") %>%
        update_role(all_nominal(), -all_outcomes(), new_role = "predictor") %>%
        update_role(all_numeric(), -all_outcomes(), new_role = "predictor") %>%
        step_rm(dplyr::any_of(all_drops)) %>%
        step_mutate(Check_In_Time_Minutes = (lubridate::hour(lubridate::hms(Check_In_Time)) * 60 +
            lubridate::minute(lubridate::hms(Check_In_Time)))) %>%
        step_rm(Check_In_Time) %>%
        step_rm(dplyr::any_of(c("Check_In_Date", "Semester_Date", "Expected_Graduation_Date"))) %>%
        step_impute_mean(all_numeric_predictors()) %>%
        step_novel(all_nominal_predictors()) %>%
        step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
        step_zv(all_predictors()) %>%
        step_normalize(all_numeric_predictors())

    return(rec)
}
```

## Python Framework Preprocessing

For the neural network models (MLP, GRU) implemented in Python using PyTorch, the preprocessing started with the output of the R feature engineering process (typically loaded as CSV or parquet files into pandas DataFrames).

Key Python preprocessing steps included:

- **Column Dropping**: Removing identifier columns or other raw features not directly used by the models.
- **Column Alignment**: Ensuring consistency in the feature set between training, validation, and holdout splits. Since dummy variables were created in the R `recipe` step, this involved adding columns with zero values if they were present in the training set but not a validation/holdout fold, and removing columns present in validation/holdout but not training.
- **Boolean Conversion**: Converting any remaining boolean columns (often resulting from dummy encoding) to integer format (0 or 1).
- **Feature Scaling**: Applying `sklearn.preprocessing.StandardScaler` to all numeric features. Crucially, the scaler was *fit only* on the training portion of the data (the 75% used for CV) and then used to *transform* the training, validation, and holdout sets.

These steps prepared the data specifically for ingestion by the PyTorch models and DataLoaders. 