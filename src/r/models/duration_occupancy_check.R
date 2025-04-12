library(here)
library(dplyr)
library(lubridate)
library(hms)
library(ggplot2)

#' Prepare dates in the dataframe
#' 
#' @param df A dataframe containing Check_In_Date and Check_In_Time columns
#' @return A dataframe with processed date columns
#' @throws Error if required columns are missing
prepare_dates <- function(df) {
  required_cols <- c("Check_In_Date", "Check_In_Time", "Predicted_Duration")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns: ", 
         paste(setdiff(required_cols, names(df)), collapse = ", "))
  }
  
  tryCatch({
    df %>% mutate(
      Check_In_Date = ymd(Check_In_Date),
      Check_In_Time = hms::as_hms(Check_In_Time),
      Check_Out_Time = hms::as_hms(as.numeric(Check_In_Time) + 
                                    round(Predicted_Duration) * 60)
    )
  }, error = function(e) {
    stop("Error processing dates: ", e$message)
  })
}

#' Calculate cumulative arrivals and departures
#' 
#' @param df A dataframe with prepared dates
#' @return A dataframe with cumulative metrics
calculate_cumulative_metrics <- function(df) {
  if (!all(c("Check_In_Date", "Check_In_Time", "Check_Out_Time") %in% names(df))) {
    stop("Missing required columns for cumulative metrics calculation")
  }
  
  df %>%
    mutate(original_order = row_number()) %>%
    arrange(Check_In_Date, Check_In_Time) %>%
    group_by(Check_In_Date) %>%
    mutate(
      Cum_Arrivals = row_number(),
      Cum_Departures = vapply(seq_along(Check_In_Time), 
                             function(i) {
                               sum(!is.na(Check_Out_Time[1:i]) & 
                                    Check_Out_Time[1:i] <= Check_In_Time[i])
                             }, FUN.VALUE = numeric(1))
    ) %>%
    ungroup() %>%
    arrange(original_order) %>%
    select(-original_order)
}

#' Calculate occupancy from cumulative metrics
#' 
#' @param df A dataframe with cumulative metrics
#' @return A dataframe with occupancy calculations
calculate_occupancy <- function(df) {
  if (!all(c("Cum_Arrivals", "Cum_Departures") %in% names(df))) {
    stop("Missing required columns for occupancy calculation")
  }
  
  df %>%
    mutate(Occupancy = Cum_Arrivals - Cum_Departures) %>%
    select(-c(Cum_Arrivals, Cum_Departures))
}

#' Process occupancy predictions from input file
#' 
#' @param input_file Path to the CSV file containing predictions
#' @return A dataframe with processed occupancy data
#' @throws Error if file cannot be read or processing fails
process_occupancy_predictions <- function(input_file) {
  if (!file.exists(input_file)) {
    stop("Input file does not exist: ", input_file)
  }
  
  tryCatch({
    read.csv(input_file) %>%
      prepare_dates() %>%
      select(Check_In_Date, Check_In_Time, Check_Out_Time) %>%
      calculate_cumulative_metrics() %>%
      calculate_occupancy()
  }, error = function(e) {
    stop("Error processing occupancy predictions: ", e$message)
  })
}

#' Plot occupancy over time
#' 
#' @param df A dataframe with occupancy data
#' @return A ggplot object
#' @throws Error if required columns are missing
plot_occupancy <- function(df) {
  required_cols <- c("Check_In_Time", "Occupancy", "Check_In_Date")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns for plotting")
  }
  
  ggplot(df, aes(x = Check_In_Time, y = Occupancy, color = Check_In_Date)) +
    geom_line() +
    geom_point() +
    theme_bw() +
    labs(
      title = "Occupancy Throughout the Day",
      x = "Time of Day",
      y = "Number of Occupants"
    ) +
    scale_x_time(breaks = hours(seq(0, 24, by = 2)),
                 labels = function(x) format(x, format = "%H:%M"))
}

#' Calculate occupancy statistics
#' 
#' @param df A dataframe with occupancy data
#' @return A dataframe with summary statistics
#' @throws Error if required columns are missing
summarize_occupancy <- function(df) {
  required_cols <- c("Check_In_Date", "Occupancy")
  if (!all(required_cols %in% names(df))) {
    stop("Missing required columns for summary")
  }
  
  df %>%
    group_by(Check_In_Date) %>%
    summarise(
      max_occupancy = max(Occupancy),
      min_occupancy = min(Occupancy),
      mean_occupancy = mean(Occupancy),
      sd_occupancy = sd(Occupancy)
    )
}

#' Plot occupancy histogram comparing predicted vs actual
#' 
#' @param inferred_from_duration Dataframe with predicted occupancy
#' @param predicted_from_psplines Dataframe with actual occupancy
#' @return A ggplot object
#' @throws Error if required columns are missing
plot_occupancy_histogram <- function(inferred_from_duration, predicted_from_psplines) {
  if (!all(c("Occupancy") %in% names(inferred_from_duration)) || 
      !all(c("Occupancy") %in% names(predicted_from_psplines))) {
    stop("Missing Occupancy column in input dataframes")
  }
  
  # Combine the data
  inferred_from_duration <- inferred_from_duration %>% mutate(Type = "Imputed")
  predicted_from_psplines <- predicted_from_psplines %>% mutate(Type = "P-Splines")
  
  combined_df <- bind_rows(inferred_from_duration, predicted_from_psplines)
  
  # Calculate means and medians for each group
  summary_stats <- combined_df %>%
    group_by(Type) %>%
    summarise(
      mean_occ = mean(Occupancy),
      median_occ = median(Occupancy)
    )
  
  # Create annotation data frames with different y positions for each type
  max_count <- max(combined_df %>% count(Occupancy) %>% pull(n))
  mean_labels <- summary_stats %>%
    mutate(
      label = sprintf("Mean: %.1f", mean_occ),
      y = case_when(
        Type == "Imputed" ~ max_count * 0.35,
        Type == "P-Splines" ~ max_count * 0.30
      )
    )
  
  median_labels <- summary_stats %>%
    mutate(
      label = sprintf("Median: %.1f", median_occ),
      y = case_when(
        Type == "Imputed" ~ max_count * 0.25,
        Type == "P-Splines" ~ max_count * 0.20
      )
    )
  
  # Define darker colors for the text
  darker_colors <- c(
    "Imputed" = "#00008B",     # Darker blue
    "P-Splines" = "#8B0000"    # Darker red
  )
  
  ggplot(combined_df, aes(x = Occupancy, fill = Type)) +
    geom_histogram(position = position_identity(),
                  binwidth = 1, 
                  alpha = 0.5) +
    # # Add mean lines
    # geom_vline(data = summary_stats,
    #            aes(xintercept = mean_occ, color = Type),
    #            linetype = "dashed",
    #            size = 1) +
    # # Add median lines
    # geom_vline(data = summary_stats,
    #            aes(xintercept = median_occ, color = Type),
    #            linetype = "solid",
    #            size = 1) +
    # # Add mean labels with darker colors
    # geom_text(data = mean_labels,
    #           aes(x = mean_occ, y = y, label = label),
    #           color = darker_colors[mean_labels$Type],
    #           hjust = -0.1,
    #           vjust = 0,
    #           fontface = "bold") +
    # # Add median labels with darker colors
    # geom_text(data = median_labels,
    #           aes(x = median_occ, y = y, label = label),
    #           color = darker_colors[median_labels$Type],
    #           hjust = -0.1,
    #           vjust = 0,
    #           fontface = "bold") +
    theme_bw() +
    labs(
      title = "Distribution of Occupancy: Imputed (from Duration) vs Directly Modeled",
      x = "Number of Occupants",
      y = "Count",
    #   caption = "Dashed lines: means; Solid lines: medians"
    )
}

# Example usage in a main function
main <- function() {
  input_file <- "results/predictions/duration_predictions.csv"
  
  # Process occupancy
  occupancy_df <- process_occupancy_predictions(input_file)
  
  # Read the actual occupancy data
  actual_occupancy <- read.csv("output/occupancy.csv")
  
  # Create and save visualizations
#   occupancy_plot <- plot_occupancy(occupancy_df)
  occupancy_histogram <- plot_occupancy_histogram(occupancy_df, actual_occupancy)
  
  # Save the histogram
#   ggsave("presentation/images/evaluation/occupancy_comparison_histogram.jpg", 
#          occupancy_histogram, 
#          width = 10, 
#          height = 6,
#          dpi = 300)
  
  # Generate and save summary statistics
  occupancy_summary <- summarize_occupancy(occupancy_df)
  
  return(list(
    occupancy_data = occupancy_df,
    summary = occupancy_summary,
    # plot = occupancy_plot,
    histogram = occupancy_histogram
  ))
}

# Assuming you have two data frames: original_data and predicted_data
# Both should have columns: Duration_In_Min and Occupancy

# Only run main if this script is being run directly
if (sys.nframe() == 0) {
  main()
}

