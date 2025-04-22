# Load required packages
# suppressPackageStartupMessages(library(config)) # No longer relying on config::get()
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(yaml))
suppressPackageStartupMessages(library(purrr)) # Needed for list manipulation

#' Recursively merge two lists
#'
#' Overwrites values in list `a` with values in list `b`.
#' Recurses for named list elements.
#' @param a Base list
#' @param b Overlay list
#' @return Merged list
recursive_merge <- function(a, b) {
    if (!is.list(a) || !is.list(b)) {
        stop("Both arguments must be lists")
    }

    nms_a <- names(a)
    nms_b <- names(b)

    # If either list is unnamed, cannot merge recursively in a meaningful way
    if (is.null(nms_a) || is.null(nms_b)) {
        # Simple overwrite: elements from b take precedence
        # This handles cases like merging vectors or unnamed lists
        # Combine and take unique names
        all_elements <- c(a, b)
        # Prioritize elements from b in case of overlap (this might not be ideal for all unnamed cases)
        # A simpler approach for unnamed might be just `return(b)` if overlay is desired.
        # For named lists where *some* elements are unnamed, this is tricky.
        # Let's stick to named list merging primarily.
        warning("Attempting merge with unnamed lists or elements. Behavior might be unexpected. Prioritizing elements from the second list.")
        # A very basic merge for potentially mixed named/unnamed
        merged_list <- a
        for (name in nms_b) {
            merged_list[[name]] <- b[[name]]
        }
        return(merged_list)
    }

    # Ensure all names are unique within each list (should be true for valid YAML)
    if (any(duplicated(nms_a)) || any(duplicated(nms_b))) {
        warning("Duplicate names found within lists being merged.")
    }

    # Get all unique names from both lists
    all_nms <- union(nms_a, nms_b)

    merged_list <- list()

    for (nm in all_nms) {
        in_a <- nm %in% nms_a
        in_b <- nm %in% nms_b

        val_a <- if (in_a) a[[nm]] else NULL
        val_b <- if (in_b) b[[nm]] else NULL

        if (in_a && in_b && is.list(val_a) && is.list(val_b) && !is.null(names(val_a)) && !is.null(names(val_b))) {
            # Both are named lists, recurse
            merged_list[[nm]] <- recursive_merge(val_a, val_b)
        } else if (in_b) {
            # Value only in b, or b's value takes precedence (not a list or a is not list)
            merged_list[[nm]] <- val_b
        } else {
            # Value only in a
            merged_list[[nm]] <- val_a
        }
    }

    return(merged_list)
}


#' Load configuration based on environment by manually reading and merging YAML files
#'
#' @param config_dir Directory containing config files (relative to project root)
#' @param env Environment to load (default: Sys.getenv("R_CONFIG_ACTIVE", "default"))
#' @return Configuration object (merged list)
#' @export
load_config <- function(config_dir = "config",
                        env = Sys.getenv("R_CONFIG_ACTIVE", "default")) {
    # Get absolute config path
    config_path <- normalizePath(here::here(config_dir), mustWork = FALSE)
    message("Looking for config in: ", config_path)
    if (!dir.exists(config_path)) stop("Configuration directory not found: ", config_path)

    # Config file names
    main_config_file <- "config.yml"
    main_config_path <- file.path(config_path, main_config_file)

    if (!file.exists(main_config_path)) {
        stop("Main configuration file not found: ", main_config_path)
    }

    # Temporarily change WD for relative path resolution within YAML files
    original_wd <- getwd()
    setwd(config_path)
    message("Temporarily changed WD to: ", config_path)

    merged_config <- list()
    load_error <- NULL

    tryCatch({
        # 1. Read the main config.yml
        message("Reading main config file: ", main_config_file)
        main_config_content <- yaml::read_yaml(main_config_file)

        # 2. Determine the active environment config section
        if (!env %in% names(main_config_content)) {
            stop("Environment '", env, "' not found in ", main_config_file)
        }
        active_env_config <- main_config_content[[env]]

        # 3. Get the list of files to inherit for the active environment
        files_to_inherit <- active_env_config$inherits
        if (is.null(files_to_inherit)) {
            stop("Environment '", env, "' in ", main_config_file, " must have an 'inherits' field.")
        }
        # Ensure it's a list/vector, even if only one file
        if (!is.list(files_to_inherit) && !is.vector(files_to_inherit)) {
            files_to_inherit <- list(files_to_inherit)
        }

        message("Environment '", env, "' inherits from: ", paste(files_to_inherit, collapse = ", "))

        # 4. Read and merge inherited files in order
        # Initialize with an empty list
        current_merged_config <- list()

        for (file_to_inherit in files_to_inherit) {
            if (!is.character(file_to_inherit) || nchar(file_to_inherit) == 0) {
                warning("Invalid filename found in inherits list: ", file_to_inherit)
                next
            }
            inherit_file_path <- file.path(config_path, file_to_inherit) # Use full path? No, relative to current WD (config_path)

            if (!file.exists(file_to_inherit)) { # Check relative path
                stop("Inherited config file not found: ", file_to_inherit, " (checked relative to ", config_path, ")")
            }

            message("Reading inherited file: ", file_to_inherit)
            inherited_content <- yaml::read_yaml(file_to_inherit)

            message("Merging content from: ", file_to_inherit)
            current_merged_config <- recursive_merge(current_merged_config, inherited_content)
        }

        # 5. Merge the specific values from the active environment section (if any, besides 'inherits')
        # Remove 'inherits' before merging the active environment's direct overrides
        active_env_overrides <- active_env_config
        active_env_overrides$inherits <- NULL

        if (length(active_env_overrides) > 0) {
            message("Merging specific overrides for environment '", env, "'")
            merged_config <- recursive_merge(current_merged_config, active_env_overrides)
        } else {
            merged_config <- current_merged_config
        }

        message("Successfully loaded and manually merged configuration for environment: ", env)
    }, error = function(e) {
        load_error <<- e
    }, finally = {
        setwd(original_wd)
        message("Restored WD to: ", original_wd)
    })

    # Check for errors during loading/merging
    if (!is.null(load_error)) {
        stop("Fatal Error during manual config loading/merging: ", load_error$message)
    }
    if (length(merged_config) == 0) {
        warning("Manual config merging resulted in an empty configuration.")
    }

    # Return the final merged list
    return(merged_config)
}


#' Get configuration value with dot notation
#' (This should work fine with the manually merged list)
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
            return(default)
        }
        result <- result[[part]]
    }

    return(result)
}
