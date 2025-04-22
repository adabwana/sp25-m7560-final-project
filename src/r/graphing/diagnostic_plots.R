# src/r/graphing/diagnostic_plots.R

# --- Load Libraries ---
# Using suppressPackageStartupMessages to minimize console noise
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(patchwork)) # For combining plots
suppressPackageStartupMessages(library(tune)) # For collect_predictions
suppressPackageStartupMessages(library(glue))

# --- Set Global Theme for Plots ---
ggplot2::theme_set(theme_minimal())

#' Generate Standard Diagnostic Plots for Regression Models
#'
#' Creates Predicted vs. Actual, Residuals vs. Predicted, Histogram of Residuals,
#' and QQ Plot of Residuals based on the output of tidymodels' last_fit().
#'
#' @param last_fit_result The result object from tune::last_fit().
#'                        This object contains the predictions on the holdout set.
#' @param target_var_name The character string name of the target variable (the actual/truth column).
#' @param output_dir Optional. A directory path to save the combined plot.
#'                   If NULL, the combined plot object is returned.
#' @param plot_filename Optional. The filename for the saved plot (if output_dir is provided).
#'                      Defaults to "diagnostic_plots.png".
#'
#' @return If output_dir is NULL, returns a patchwork ggplot object.
#'         Otherwise, saves the plot to the specified file and returns NULL invisibly.
#' @export
#'
#' @examples
#' # Assuming 'best_fit_result' is an object saved from last_fit()
#' # saved_plots <- generate_diagnostic_plots(best_fit_result)
#' # print(saved_plots)
#'
#' # Or save directly to a file
#' # generate_diagnostic_plots(best_fit_result, target_var_name = "YourTarget", output_dir = "artifacts/r/plots")
generate_diagnostic_plots <- function(last_fit_result, target_var_name, output_dir = NULL, plot_filename = "diagnostic_plots.png") {
    # --- Argument Validation ---
    if (!inherits(last_fit_result, "last_fit")) {
        stop("Input must be a 'last_fit' object produced by tune::last_fit().")
    }
    if (missing(target_var_name) || !is.character(target_var_name) || length(target_var_name) != 1) {
        stop("Argument 'target_var_name' must be provided as a single character string.")
    }
    if (!is.null(output_dir) && !dir.exists(output_dir)) {
        warning(glue::glue("Output directory '{output_dir}' does not exist. Creating it."))
        dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }

    # --- Extract Predictions and Calculate Residuals ---
    cat("--- Extracting predictions from last_fit result ---\n")
    predictions_df <- tune::collect_predictions(last_fit_result)

    # Check if the provided target variable name exists in the predictions
    if (!target_var_name %in% names(predictions_df)) {
        stop(glue::glue("The specified target variable '{target_var_name}' was not found in the columns of collect_predictions(). Available columns: {paste(names(predictions_df), collapse=', ')}"))
    }
    # Remove the guessing logic
    # potential_truth_cols <- setdiff(names(predictions_df), c(".pred", ".row", ".config"))
    # if (length(potential_truth_cols) != 1) {
    #     stop("Could not reliably determine the truth column in predictions_df. Expected one column besides .pred, .row, .config.")
    # }
    # truth_col_name <- potential_truth_cols[1]
    # cat(glue::glue("Identified truth column: {truth_col_name}\n"))

    # Use the provided target_var_name
    cat(glue::glue("Using specified truth column: {target_var_name}\n"))
    truth_col_name <- target_var_name

    plot_data <- predictions_df %>%
        dplyr::select(truth = !!rlang::sym(truth_col_name), predicted = .pred) %>%
        dplyr::mutate(residuals = truth - predicted)

    # --- Generate Plots ---
    cat("--- Generating diagnostic plots ---\n")

    # Calculate combined range for symmetrical axes in p1
    min_val <- min(c(plot_data$truth, plot_data$predicted), na.rm = TRUE)
    max_val <- max(c(plot_data$truth, plot_data$predicted), na.rm = TRUE)
    axis_limits <- c(min_val, max_val)

    # 1. Predicted vs Actual
    p1 <- ggplot(plot_data, aes(x = truth, y = predicted)) +
        geom_point(alpha = 0.6, color = "steelblue") +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
        labs(
            title = "Predicted vs. Actual (Holdout Set)",
            x = "Actual Values",
            y = "Predicted Values"
        ) +
        # coord_fixed() + # Ensures 1:1 aspect ratio for the abline
        # Set symmetrical axis limits using coord_cartesian
        coord_cartesian(xlim = axis_limits, ylim = axis_limits)

    # 2. Residuals vs Predicted
    p2 <- ggplot(plot_data, aes(x = predicted, y = residuals)) +
        geom_point(alpha = 0.6, color = "steelblue") +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        labs(
            title = "Residuals vs. Predicted",
            x = "Predicted Values",
            y = "Residuals"
        )

    # 3. Histogram of Residuals
    p3 <- ggplot(plot_data, aes(x = residuals)) +
        geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", color = "black") +
        # geom_density(color = "steelblue") +
        labs(
            title = "Histogram of Residuals",
            x = "Residuals",
            y = "Density"
        )

    # 4. QQ Plot of Residuals
    p4 <- ggplot(plot_data, aes(sample = residuals)) +
        stat_qq(color = "steelblue") +
        stat_qq_line(color = "red", linetype = "dashed") +
        labs(
            title = "Normal Q-Q Plot of Residuals",
            x = "Theoretical Quantiles",
            y = "Sample Quantiles (Residuals)"
        )

    # --- Combine Plots ---
    combined_plot <- (p1 + p2) / (p3 + p4) +
        patchwork::plot_annotation(
            title = "Regression Model Diagnostics (Holdout Set)",
            theme = theme(plot.title = element_text(hjust = 0.5))
        )

    # --- Save or Return ---
    if (!is.null(output_dir)) {
        save_path <- file.path(output_dir, plot_filename)
        cat(glue::glue("--- Saving combined diagnostic plot to {save_path} ---\n"))
        ggsave(save_path, plot = combined_plot, width = 10, height = 8, dpi = 300, bg = "white")
        return(invisible(NULL))
    } else {
        cat("--- Returning combined plot object ---\n")
        return(combined_plot)
    }
}
