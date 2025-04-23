# Custom yardstick metrics for integer predictions

suppressPackageStartupMessages(library(yardstick))
suppressPackageStartupMessages(library(rlang))

# ---------------------- rmse_int ----------------------

# Internal implementation: Calculates RMSE on already rounded estimate
rmse_int_impl <- function(truth, estimate, case_weights = NULL) {
    # The core calculation is just standard RMSE
    # We assume estimate is already rounded before calling this
    rmse_vec(truth = truth, estimate = estimate, na_rm = FALSE, case_weights = case_weights)
}

# Vector version: Handles checks, NA, rounding, calls impl
rmse_int_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    check_numeric_metric(truth, estimate, case_weights)

    if (na_rm) {
        result <- yardstick_remove_missing(truth, estimate, case_weights)
        truth <- result$truth
        estimate <- result$estimate
        case_weights <- result$case_weights
    } else if (yardstick_any_missing(truth, estimate, case_weights)) {
        return(NA_real_)
    }

    # Round the estimate before calculating
    estimate_rounded <- round(estimate)

    rmse_int_impl(truth = truth, estimate = estimate_rounded, case_weights = case_weights)
}

# Generic function definition
rmse_int <- function(data, ...) {
    UseMethod("rmse_int")
}

# Register as a numeric metric
rmse_int <- new_numeric_metric(rmse_int, direction = "minimize")

# Data frame method
rmse_int.data.frame <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    numeric_metric_summarizer(
        name = "rmse_int",
        fn = rmse_int_vec,
        data = data,
        truth = !!enquo(truth),
        estimate = !!enquo(estimate),
        na_rm = na_rm,
        case_weights = !!enquo(case_weights)
    )
}

# Grouped data frame method (calls the data.frame method)
rmse_int.grouped_df <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    rmse_int.data.frame(data, !!enquo(truth), !!enquo(estimate), na_rm, !!enquo(case_weights), ...)
}


# ---------------------- mae_int ----------------------

# Internal implementation: Calculates MAE on already rounded estimate
mae_int_impl <- function(truth, estimate, case_weights = NULL) {
    mae_vec(truth = truth, estimate = estimate, na_rm = FALSE, case_weights = case_weights)
}

# Vector version: Handles checks, NA, rounding, calls impl
mae_int_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    check_numeric_metric(truth, estimate, case_weights)

    if (na_rm) {
        result <- yardstick_remove_missing(truth, estimate, case_weights)
        truth <- result$truth
        estimate <- result$estimate
        case_weights <- result$case_weights
    } else if (yardstick_any_missing(truth, estimate, case_weights)) {
        return(NA_real_)
    }

    estimate_rounded <- round(estimate)
    mae_int_impl(truth = truth, estimate = estimate_rounded, case_weights = case_weights)
}

# Generic function definition
mae_int <- function(data, ...) {
    UseMethod("mae_int")
}

# Register as a numeric metric
mae_int <- new_numeric_metric(mae_int, direction = "minimize")

# Data frame method
mae_int.data.frame <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    numeric_metric_summarizer(
        name = "mae_int",
        fn = mae_int_vec,
        data = data,
        truth = !!enquo(truth),
        estimate = !!enquo(estimate),
        na_rm = na_rm,
        case_weights = !!enquo(case_weights)
    )
}

# Grouped data frame method (calls the data.frame method)
mae_int.grouped_df <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    mae_int.data.frame(data, !!enquo(truth), !!enquo(estimate), na_rm, !!enquo(case_weights), ...)
}


# ---------------------- rsq_int ----------------------

# Internal implementation: Calculates R-squared on already rounded estimate
rsq_int_impl <- function(truth, estimate, case_weights = NULL) {
    rsq_vec(truth = truth, estimate = estimate, na_rm = FALSE, case_weights = case_weights)
}

# Vector version: Handles checks, NA, rounding, calls impl
rsq_int_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    check_numeric_metric(truth, estimate, case_weights)

    if (na_rm) {
        result <- yardstick_remove_missing(truth, estimate, case_weights)
        truth <- result$truth
        estimate <- result$estimate
        case_weights <- result$case_weights
    } else if (yardstick_any_missing(truth, estimate, case_weights)) {
        return(NA_real_)
    }

    estimate_rounded <- round(estimate)
    rsq_int_impl(truth = truth, estimate = estimate_rounded, case_weights = case_weights)
}

# Generic function definition
rsq_int <- function(data, ...) {
    UseMethod("rsq_int")
}

# Register as a numeric metric
rsq_int <- new_numeric_metric(rsq_int, direction = "maximize")

# Data frame method
rsq_int.data.frame <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    numeric_metric_summarizer(
        name = "rsq_int",
        fn = rsq_int_vec,
        data = data,
        truth = !!enquo(truth),
        estimate = !!enquo(estimate),
        na_rm = na_rm,
        case_weights = !!enquo(case_weights)
    )
}

# Grouped data frame method (calls the data.frame method)
rsq_int.grouped_df <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
    rsq_int.data.frame(data, !!enquo(truth), !!enquo(estimate), na_rm, !!enquo(case_weights), ...)
}

# --- End of Custom Metrics ---
cat("--- Custom integer metrics (rmse_int, mae_int, rsq_int) loaded successfully ---\n")
