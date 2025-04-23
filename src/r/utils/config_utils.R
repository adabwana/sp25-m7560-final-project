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
#' @param override_value If not NULL, this value is returned directly, overriding any config value.
#' @param default Default value to return if the path is not found in the config AND override_value is NULL.
#' @return Configuration value or override value or default value.
#' @export
get_config_value <- function(config, path, override_value = NULL, default = NULL) {
    # 1. Check if an override value is provided
    if (!is.null(override_value)) {
        return(override_value)
    }

    # 2. If no override, proceed to look up in config
    parts <- strsplit(path, "\\.")[[1]]
    result <- config
    for (part in parts) {
        # Check if result is a list and the part exists
        if (!is.list(result) || !part %in% names(result)) {
            # Path not found, return the default value
            return(default)
        }
        result <- result[[part]]
    }

    # 3. Path found, check if the result is NULL
    # If the config explicitly has NULL for this path, and a default is specified,
    # should we return the default or the NULL? Current logic returns default.
    # Let's stick to that for consistency unless specified otherwise.
    if (is.null(result) && !is.null(default)) {
        return(default)
    }

    # 4. Path found and value is not NULL (or it is NULL and no default was given)
    return(result)
}
