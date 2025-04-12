library(dplyr)
library(ggplot2)
library(gridExtra)

#' Read and prepare data for comparison
#' 
#' @return List containing prepared dataframes for duration and occupancy
prepare_comparison_data <- function() {
  # Read real data
  real_data <- read.csv("data/processed/train_engineered.csv")
  
  # Read predicted data
  predicted_duration <- read.csv("output/duration.csv")
  predicted_occupancy <- read.csv("output/occupancy.csv")
  
  # Rename columns for consistency
  colnames(predicted_duration) <- "Duration_In_Min"
  colnames(predicted_occupancy) <- "Occupancy"
  
  return(list(
    real = real_data,
    predicted_duration = predicted_duration,
    predicted_occupancy = predicted_occupancy
  ))
}

#' Create side-by-side distribution comparison plots
#' 
#' @param data List containing the prepared dataframes
#' @return A ggplot object with side-by-side histograms
plot_distributions <- function(data) {
  # Duration plot
  p1 <- ggplot() +
    geom_histogram(data = data$real,
                  aes(x = Duration_In_Min, fill = "Real"),
                  alpha = 0.7, binwidth = 5) +
    geom_histogram(data = data$predicted_duration,
                  aes(x = Duration_In_Min, fill = "Predicted"),
                  alpha = 0.7, binwidth = 5) +
    theme_bw() +
    labs(title = "Duration Distribution",
         x = "Duration (minutes)",
         y = "Count",
         fill = "Type") +
    scale_fill_manual(values = c("Real" = "gray80", "Predicted" = "#00BFC4")) +
    theme(legend.position = "none")  # Remove legend from the first plot
  
  # Occupancy plot
  p2 <- ggplot() +
    geom_histogram(data = data$real,
                  aes(x = Occupancy, fill = "Real"),
                  alpha = 0.7, binwidth = 1) +
    geom_histogram(data = data$predicted_occupancy,
                  aes(x = Occupancy, fill = "Predicted"),
                  alpha = 0.7, binwidth = 1) +
    theme_bw() +
    labs(title = "Occupancy Distribution",
         x = "Number of Occupants",
         y = "Count",
         fill = "Type") +
    scale_fill_manual(values = c("Real" = "gray80", "Predicted" = "#00BFC4")) +
    theme(legend.position = "right")  # Move legend to the right
  
  # Combine plots side by side
  combined_plot <- grid.arrange(p1, p2, ncol = 2)
  
  return(combined_plot)
}

main <- function() {
  # Prepare data
  data <- prepare_comparison_data()
  
  # Create plots
  comparison_plots <- plot_distributions(data)
  
  # Save the plot
  ggsave("presentation/images/evaluation/distribution_comparisons.jpg", 
         comparison_plots, 
         width = 15, 
         height = 6,
         dpi = 300)
  
  return(comparison_plots)
}

# Run main if script is run directly
if (sys.nframe() == 0) {
  main()
} 