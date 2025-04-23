# src/r/graphing/duration_occupancy_pred_plots.R

# --- Load Libraries ---
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(glue))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(hms))

# --- Configuration ---
PLOTS_DIR <- here::here("artifacts", "r", "plots")
TRAINING_DATA_PATH <- here::here("data", "processed", "train_engineered.csv")
TEST_DATA_PATH <- here::here("data", "processed", "test_engineered.csv")

# Create plots directory if it doesn't exist
dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

#' Read the latest prediction file for a target variable
#'
#' @param target_var Target variable name (e.g., "Duration_In_Min" or "Occupancy")
#' @return Path to the latest prediction file
get_latest_pred_file <- function(target_var) {
  predictions_dir <- here::here("data", "predictions")
  all_files <- list.files(predictions_dir, pattern = glue("{target_var}.*\\.csv$"), full.names = TRUE)
  
  if (length(all_files) == 0) {
    stop(glue("No prediction files found for {target_var}"))
  }
  
  # Sort by modification time and get the most recent
  file_info <- file.info(all_files)
  most_recent <- rownames(file_info)[which.max(file.info(all_files)$mtime)]
  
  cat(glue("Using prediction file: {basename(most_recent)}\n"))
  return(most_recent)
}

#' Load the training data with selected columns
#'
#' @param cols Columns to select
#' @return Dataframe with selected columns
load_training_data <- function(cols = c()) {
  cat("Loading training data...\n")
  train_data <- read_csv(TRAINING_DATA_PATH, show_col_types = FALSE)
  
  if (length(cols) > 0) {
    train_data <- train_data %>% select(all_of(cols))
  }
  
  return(train_data)
}

#' Load the test data with selected columns
#'
#' @param cols Columns to select
#' @return Dataframe with selected columns
load_test_data <- function(cols = c()) {
  cat("Loading test data...\n")
  test_data <- read_csv(TEST_DATA_PATH, show_col_types = FALSE)
  
  if (length(cols) > 0) {
    train_data <- train_data %>% select(all_of(cols))
  }
  
  return(test_data)
}

#' Load prediction data for a target variable
#'
#' @param target_var Target variable name
#' @return Dataframe with predictions
load_prediction_data <- function(target_var) {
  pred_file <- get_latest_pred_file(target_var)
  cat(glue("Loading {target_var} predictions...\n"))
  pred_data <- read_csv(pred_file, show_col_types = FALSE)
  return(pred_data)
}

#' Prepare data for histogram comparison
#'
#' @param actual_data Dataframe with actual values
#' @param predicted_data Dataframe with predicted values
#' @param actual_col Column name for actual values
#' @param pred_col Column name for predicted values
#' @return Dataframe prepared for plotting
prepare_histogram_data <- function(actual_data, predicted_data, actual_col, pred_col = ".pred") {
  # Create dataframes with source labels
  actual_df <- actual_data %>%
    select(all_of(actual_col)) %>%
    rename(value = all_of(actual_col)) %>%
    mutate(source = "Actual")
  
  pred_df <- predicted_data %>%
    select(all_of(pred_col)) %>%
    rename(value = all_of(pred_col)) %>%
    mutate(source = "Predicted")
  
  # Combine the dataframes
  bind_rows(actual_df, pred_df)
}

#' Plot histogram comparing actual vs predicted values
#'
#' @param data Prepared dataframe with value and source columns
#' @param title Plot title
#' @param x_label X-axis label
#' @param binwidth Bin width for histogram
#' @param file_name Output file name
#' @return ggplot object
plot_comparison_histogram <- function(data, title, x_label, binwidth = NULL, file_name = NULL) {
  # Calculate summary statistics
  summary_stats <- data %>%
    group_by(source) %>%
    summarise(
      mean_val = mean(value, na.rm = TRUE),
      median_val = median(value, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Determine binwidth if not provided
  if (is.null(binwidth)) {
    # Use Freedman-Diaconis rule if the variable is continuous
    if (length(unique(data$value)) > 30) {
      iqr <- IQR(data$value, na.rm = TRUE)
      n <- nrow(data)
      binwidth <- 2 * iqr / (n^(1/3))
      # If binwidth is too small, use a reasonable default
      if (binwidth < 0.1) binwidth <- 1
    } else {
      # For discrete variables, use 1
      binwidth <- 1
    }
  }
  
  # Create the plot
  p <- ggplot(data, aes(x = value, fill = source)) +
    geom_histogram(position = "identity", alpha = 0.6, binwidth = binwidth) +
    geom_vline(data = summary_stats, 
               aes(xintercept = mean_val, color = source),
               linetype = "dashed", linewidth = 1) +
    geom_vline(data = summary_stats, 
               aes(xintercept = median_val, color = source),
               linetype = "solid", linewidth = 1) +
    labs(
      title = title,
      x = x_label,
      y = "Count",
      caption = "Dashed lines: means; Solid lines: medians"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.title = element_blank(),
      legend.position = "top"
    )
  
  # Save the plot if filename is provided
  if (!is.null(file_name)) {
    save_path <- file.path(PLOTS_DIR, file_name)
    cat(glue("Saving plot to {save_path}\n"))
    ggsave(save_path, p, width = 10, height = 6, dpi = 300, bg = "white")
  }
  
  return(p)
}

#' Calculate occupancy from duration predictions using a simpler, more robust method
#'
#' This function simulates occupancy based on duration predictions by:
#' 1. Creating a timeline for each day
#' 2. For each check-in, determining when the person leaves based on duration
#' 3. Counting occupants at each point in time
#'
#' @param duration_preds Dataframe with duration predictions (.pred column)
#' @param train_data Training data with check-in times and dates
#' @return Dataframe with imputed occupancy values
calculate_imputed_occupancy <- function(duration_preds, train_data) {
  cat("Calculating imputed occupancy from duration predictions...\n")
  
  # Sample size checking and alignment
  if (nrow(train_data) != nrow(duration_preds)) {
    cat(glue("WARNING: Row count mismatch - training data: {nrow(train_data)}, predictions: {nrow(duration_preds)}\n"))
    sample_size <- min(nrow(train_data), nrow(duration_preds))
    cat(glue("Using first {sample_size} rows for both datasets\n"))
    
    train_data <- train_data[1:sample_size, ]
    duration_preds <- duration_preds[1:sample_size, ]
  }
  
  # Extract needed columns and combine with predictions
  session_data <- train_data %>%
    select(Check_In_Date, Check_In_Time) %>%
    bind_cols(Predicted_Duration = duration_preds$.pred)
  
  # Make sure we're working with the right data types
  session_data <- session_data %>%
    mutate(
      # Convert date strings to Date objects
      Check_In_Date = as.Date(Check_In_Date),
      
      # Store original time format for reference
      Original_Time = Check_In_Time,
      
      # Convert time to minutes since midnight for easier calculations
      # Handle both string times (HH:MM:SS) and numeric times (seconds since midnight)
      Time_Minutes = case_when(
        grepl(":", Check_In_Time) ~ {
          parts <- as.numeric(unlist(strsplit(as.character(Check_In_Time), ":")))
          if(length(parts) >= 2) {
            parts[1] * 60 + parts[2] + ifelse(length(parts) > 2, parts[3]/60, 0)
          } else {
            as.numeric(Check_In_Time) / 60
          }
        },
        TRUE ~ as.numeric(Check_In_Time) / 60
      ),
      
      # Calculate departure time in minutes
      Departure_Minutes = Time_Minutes + Predicted_Duration
    )
  
  # Handle any NAs from the time conversion
  if(sum(is.na(session_data$Time_Minutes)) > 0) {
    cat(glue("WARNING: {sum(is.na(session_data$Time_Minutes))} rows had NA times and will be removed\n"))
    session_data <- session_data %>% filter(!is.na(Time_Minutes))
  }
  
  # --- Calculate occupancy over time on each day ---
  # Create result dataframe
  occupancy_df <- tibble(
    Occupancy = integer(),
    Check_In_Date = as.Date(character()),
    Check_In_Time = character()
  )
  
  # Get unique dates
  unique_dates <- unique(session_data$Check_In_Date)
  
  # For each date, calculate minute-by-minute occupancy
  for (date in unique_dates) {
    day_data <- session_data %>% 
      filter(Check_In_Date == date)
    
    # Get min and max times for the day (with some padding)
    min_time <- floor(min(day_data$Time_Minutes, na.rm = TRUE))
    max_time <- ceiling(max(day_data$Departure_Minutes, na.rm = TRUE))
    
    # For each minute in the day, calculate occupancy
    for (minute in seq(min_time, max_time)) {
      # Count people present at this minute
      count <- sum(day_data$Time_Minutes <= minute & 
                   day_data$Departure_Minutes > minute,
                   na.rm = TRUE)
      
      # Only add non-zero occupancy to the result
      if (count > 0) {
        # Convert minute to HH:MM:SS format
        hour <- floor(minute / 60)
        min_part <- minute %% 60
        time_str <- sprintf("%02d:%02d:00", hour, min_part)
        
        # Add to results
        occupancy_df <- bind_rows(
          occupancy_df,
          tibble(
            Occupancy = count,
            Check_In_Date = date,
            Check_In_Time = time_str,
            Time_Minutes = minute
          )
        )
      }
    }

  }
  
  return(occupancy_df)
}

#' Plot imputed vs directly modeled occupancy (adapted from reference code)
#'
#' @param imputed_occupancy Dataframe with occupancy imputed from duration (must contain 'Occupancy' column)
#' @param direct_occupancy Dataframe with directly predicted occupancy (must contain '.pred' column)
#' @param file_name Output file name
#' @return ggplot object
plot_imputed_vs_direct_occupancy <- function(imputed_occupancy, direct_occupancy, file_name = NULL) {
  cat("Creating imputed vs direct occupancy histogram...\n")
  
  # Validate input dataframes
  if (!"Occupancy" %in% names(imputed_occupancy)) {
    stop("Imputed occupancy dataframe must contain 'Occupancy' column.")
  }
  if (!".pred" %in% names(direct_occupancy)) {
    stop("Directly modeled occupancy dataframe must contain '.pred' column.")
  }
  
  # Prepare imputed data
  imputed_df <- imputed_occupancy %>%
    select(Occupancy) %>%
    mutate(Type = "Imputed from Duration")
  
  # Prepare directly modeled data
  direct_df <- direct_occupancy %>%
    select(Occupancy = .pred) %>% # Rename .pred to Occupancy for consistency
    mutate(Type = "Directly Modeled")
  
  # Combine both data sources
  combined_df <- bind_rows(imputed_df, direct_df)
  
  # Calculate summary statistics for each Type
  summary_stats <- combined_df %>%
    group_by(Type) %>%
    summarise(
      mean_val = mean(Occupancy, na.rm = TRUE),
      median_val = median(Occupancy, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Print summary stats for verification
  cat("Occupancy Summary Statistics:\n")
  print(summary_stats)
  
  # Create the plot
  p <- ggplot(combined_df, aes(x = Occupancy, fill = Type)) +
    geom_histogram(position = "identity", alpha = 0.5, binwidth = 1) +
    geom_vline(data = summary_stats, 
               aes(xintercept = mean_val, color = Type),
               linetype = "dashed", linewidth = 1) +
    geom_vline(data = summary_stats, 
               aes(xintercept = median_val, color = Type),
               linetype = "solid", linewidth = 1) +
    scale_fill_manual(values = c("Imputed from Duration" = "darkred", 
                                "Directly Modeled" = '#00008B')) +
    scale_color_manual(values = c("Imputed from Duration" = "darkred", 
                                 "Directly Modeled" = '#00008B')) +
    labs(
      title = "Distribution of Occupancy: Imputed vs Directly Modeled",
      x = "Number of Occupants",
      y = "Count",
      caption = "Dashed lines: means; Solid lines: medians"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.title = element_blank(),
      legend.position = "top"
    )
  
  # Save the plot if filename is provided
  if (!is.null(file_name)) {
    save_path <- file.path(PLOTS_DIR, file_name)
    cat(glue("Saving plot to {save_path}\n"))
    ggsave(save_path, p, width = 10, height = 6, dpi = 300, bg = "white")
  }
  
  return(p)
}

# --- Helper Functions for Imputed Occupancy Calculation (Based on reference code) ---

#' Prepare dates in the dataframe
#' 
#' @param df A dataframe containing Check_In_Date, Check_In_Time, and Predicted_Duration columns
#' @return A dataframe with processed date columns
#' @throws Error if required columns are missing
prepare_dates <- function(df) {
  required_cols <- c("Check_In_Date", "Check_In_Time", "Predicted_Duration")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns for prepare_dates: ", 
         paste(setdiff(required_cols, names(df)), collapse = ", "))
  }
  
  cat("Preparing dates... Converting Check_In_Time to hms format.\n")
  
  # Ensure Check_In_Time is character before converting to hms
  df <- df %>% mutate(Check_In_Time_Char = as.character(Check_In_Time))
  
  tryCatch({
    df %>% 
      mutate(
        Check_In_Date = ymd(Check_In_Date), # Use lubridate for robust date parsing
        Check_In_Time_hms = hms::as_hms(Check_In_Time_Char), # Convert character time to hms
        Check_Out_Time_hms = hms::as_hms(as.numeric(Check_In_Time_hms) + 
                                        round(Predicted_Duration) * 60) # Calculate checkout time
      ) %>%
      select(-Check_In_Time_Char) # Remove temporary column
  }, error = function(e) {
    stop("Error processing dates in prepare_dates: ", e$message)
  })
}

#' Calculate cumulative arrivals and departures
#' 
#' @param df A dataframe with prepared dates (Check_In_Date, Check_In_Time_hms, Check_Out_Time_hms)
#' @return A dataframe with cumulative metrics
calculate_cumulative_metrics <- function(df) {
  required_cols <- c("Check_In_Date", "Check_In_Time_hms", "Check_Out_Time_hms")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns for calculate_cumulative_metrics: ",
         paste(setdiff(required_cols, names(df)), collapse = ", "))
  }
  
  cat("Calculating cumulative metrics...\n")
  df %>% 
    mutate(original_order = row_number()) %>% # Preserve original order
    arrange(Check_In_Date, Check_In_Time_hms) %>%
    group_by(Check_In_Date) %>%
    mutate(
      Cum_Arrivals = row_number(),
      Cum_Departures = vapply(seq_along(Check_In_Time_hms), 
                              function(i) {
                                sum(!is.na(Check_Out_Time_hms[1:i]) & 
                                     Check_Out_Time_hms[1:i] <= Check_In_Time_hms[i])
                              }, FUN.VALUE = numeric(1))
    ) %>% 
    ungroup() %>% 
    arrange(original_order) %>% # Restore original order
    select(-original_order) # Remove helper column
}

#' Calculate occupancy from cumulative metrics
#' 
#' @param df A dataframe with cumulative metrics (Cum_Arrivals, Cum_Departures)
#' @return A dataframe with occupancy calculations
calculate_occupancy <- function(df) {
  required_cols <- c("Cum_Arrivals", "Cum_Departures")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns for calculate_occupancy: ", 
         paste(setdiff(required_cols, names(df)), collapse = ", "))
  }
  
  cat("Calculating final occupancy...\n")
  df %>% 
    mutate(Occupancy = Cum_Arrivals - Cum_Departures) # Calculate Occupancy
}

#' Calculate imputed occupancy using the helper functions
#' 
#' This is a wrapper function that takes test data and duration predictions,
#' combines them, and runs the sequence of steps to calculate occupancy.
#' 
#' @param test_data Dataframe of the test set (e.g., test_engineered.csv)
#' @param duration_preds Dataframe with duration predictions (.pred column)
#' @return Dataframe with imputed occupancy values
calculate_imputed_occupancy_from_duration <- function(test_data, duration_preds) {
  cat("Starting imputed occupancy calculation pipeline...\n")
  
  # --- 1. Combine test data and predictions --- 
  cat("Combining test data with duration predictions...\n")
  if (nrow(test_data) != nrow(duration_preds)) {
    cat(glue("WARNING: Row count mismatch - test data: {nrow(test_data)}, duration predictions: {nrow(duration_preds)}\n"))
    sample_size <- min(nrow(test_data), nrow(duration_preds))
    cat(glue("Using first {sample_size} rows for both datasets\n"))
    test_data <- test_data[1:sample_size, ]
    duration_preds <- duration_preds[1:sample_size, ]
  }
  
  # Ensure predictions column exists
  if (!".pred" %in% names(duration_preds)) {
    stop("Duration predictions dataframe must contain a '.pred' column.")
  }
  
  # Add Predicted_Duration column
  combined_data <- test_data %>% 
    bind_cols(Predicted_Duration = duration_preds$.pred)
  
  # --- 2. Run the calculation pipeline --- 
  imputed_occupancy_df <- combined_data %>% 
    prepare_dates() %>% 
    calculate_cumulative_metrics() %>% 
    calculate_occupancy()
  
  cat("Imputed occupancy calculation complete.\n")
  return(imputed_occupancy_df)
}

# --- Execute visualization routines directly when the script is sourced ---

cat("=== Starting Duration and Occupancy Visualization ===\n\n")

# --- Load Data ---
cat("Loading necessary data...\n")
train_data <- load_training_data() # Still needed for the first two histograms
test_data <- load_test_data()

# Load predictions (handle potential errors)
tryCatch({
  duration_preds <- load_prediction_data("Duration_In_Min")
}, error = function(e) {
  stop("Failed to load Duration_In_Min predictions: ", e$message)
})
tryCatch({
  occupancy_preds <- load_prediction_data("Occupancy")
}, error = function(e) {
  stop("Failed to load Occupancy predictions: ", e$message)
})

# --- Generate Plots ---

# 1. Duration Histogram: Actual (Training) vs Predicted (Test)
cat("\n--- Creating Duration Distribution Histogram ---\n")
# Note: Comparing training actuals vs test predictions might not be ideal,
# but using test actuals would require loading test_engineered again without sampling
# if row counts mismatch during prediction loading.
# For now, we use training actuals for simplicity.
duration_hist_data <- prepare_histogram_data(train_data, duration_preds, "Duration_In_Min")
duration_hist <- plot_comparison_histogram(
  duration_hist_data,
  "Distribution of Duration: Actual (Train) vs Predicted (Test)",
  "Duration (minutes)",
  binwidth = 5,
  file_name = "duration_distribution.png"
)

# 2. Occupancy Histogram: Actual (Training) vs Predicted (Test)
cat("\n--- Creating Occupancy Distribution Histogram ---\n")
occupancy_hist_data <- prepare_histogram_data(train_data, occupancy_preds, "Occupancy", pred_col = ".pred")
occupancy_hist <- plot_comparison_histogram(
  occupancy_hist_data,
  "Distribution of Occupancy: Actual (Train) vs Predicted (Test)",
  "Number of Occupants",
  binwidth = 1,
  file_name = "occupancy_distribution.png"
)

# 3. Imputed vs Direct Occupancy Histogram (using Test Data)
cat("\n--- Creating Imputed vs Direct Occupancy Histogram ---\n")
# Calculate occupancy imputed from duration predictions using the new pipeline
tryCatch({
  imputed_occupancy_df <- calculate_imputed_occupancy_from_duration(test_data, duration_preds)
}, error = function(e) {
  stop("Failed to calculate imputed occupancy: ", e$message)
})

# Plot the comparison using the updated function
imputed_vs_direct_hist <- plot_imputed_vs_direct_occupancy(
  imputed_occupancy_df, # Contains 'Occupancy' column from the pipeline
  occupancy_preds,      # Contains '.pred' column for directly modeled occupancy
  file_name = "imputed_vs_direct_occupancy.png"
)

cat("\n=== Visualization Complete ===\n")
cat(glue("Plots saved to: {PLOTS_DIR}\n"))

# Create a list of the plots for interactive usage
histograms <- list(
  duration_histogram = duration_hist,
  occupancy_histogram = occupancy_hist,
  imputed_vs_direct_histogram = imputed_vs_direct_hist
)

