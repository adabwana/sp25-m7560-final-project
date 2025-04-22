# Load required packages
# suppressPackageStartupMessages(library(config)) # No longer using config::get
suppressPackageStartupMessages(library(yaml)) # Use yaml::read_yaml
suppressPackageStartupMessages(library(here))
# suppressPackageStartupMessages(library(purrr)) # No longer needed for merge

# The recursive_merge function is no longer needed as config::get handles merging.
# recursive_merge <- function(a, b) { ... }


#' Load configuration by reading a specific YAML file.
#'
#' @param file The path to the YAML configuration file.
#'
#' @return Configuration object (list read from YAML)
#' @export
load_config <- function(file = "config/config.yml") { # Default to main config file

    # Resolve the file path to an absolute path relative to project root if not already absolute
    # Correctly check for Windows absolute path (needs quadruple backslash for regex)
    absolute_file_path <- if (startsWith(file, "/") || startsWith(file, "~") || grepl("^[A-Za-z]:\\\\", file)) {
        # Assume already absolute (Unix-like, home dir, or Windows)
        normalizePath(file, mustWork = FALSE)
    } else {
        # Assume relative to project root
        normalizePath(here::here(file), mustWork = FALSE)
    }

    if (!file.exists(absolute_file_path)) {
        stop(glue::glue("Configuration file path does not exist: {absolute_file_path} (Resolved from input: '{file}')"))
    }

    message(glue::glue("Loading configuration using yaml::read_yaml() from: {absolute_file_path}"))

    # Read the YAML file directly
    loaded_config <- tryCatch(
        {
            yaml::read_yaml(absolute_file_path)
        },
        error = function(e) {
            stop(glue::glue("Error reading YAML configuration file '{absolute_file_path}': {e$message}"))
        }
    )

    if (is.null(loaded_config) || length(loaded_config) == 0) {
        warning(glue::glue("YAML file '{absolute_file_path}' loaded as NULL or empty list."))
    } else {
        message("Configuration loaded successfully via yaml::read_yaml().")
    }

    return(loaded_config)
}


#' Get configuration value with dot notation
#' (This helper function remains useful)
#' @param config Configuration object (list)
#' @param path Dot-separated path to config value
#' @param default Default value if path not found
#' @return Configuration value
#' @export
get_config_value <- function(config, path, default = NULL) {
    # Split path into parts
    parts <- strsplit(path, "\\.")[[1]]

    # Navigate through config
    result <- config
    for (part in parts) {
        # Check if result is a list and the part exists
        if (!is.list(result) || !part %in% names(result)) {
            # If part is not found, return the default value
            return(default)
        }
        result <- result[[part]]
    }

    # Check if the final result is NULL, and return default if specified
    if (is.null(result) && !is.null(default)) {
        return(default)
    }

    return(result)
}
