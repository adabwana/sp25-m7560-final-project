# Load necessary libraries
library(readr)
library(here)
library(glue)

#' Load Data from Processed Directory
#'
#' Reads a CSV file from the 'data/processed' directory relative to the project root.
#'
#' @param filename The name of the CSV file (e.g., "train_engineered.csv")
#' @param data_dir The subdirectory within the project root containing the data. Defaults to "data/processed".
#'
#' @return A tibble containing the loaded data.
#' @export
#'
#' @examples
#' \dontrun{
#' train_data <- load_data("train_engineered.csv")
#' test_data <- load_data("test_engineered.csv")
#' }
load_data <- function(filename, data_dir = "data/processed") {
    # Construct the full path relative to the project root
    # here() ensures the path is correct regardless of the working directory
    file_path <- here::here(data_dir, filename)

    # Check if the file exists
    if (!file.exists(file_path)) {
        stop(glue::glue("File not found at path: {file_path}"))
    }

    # Read the CSV file
    data <- readr::read_csv(file_path, show_col_types = FALSE)

    return(data)
}
