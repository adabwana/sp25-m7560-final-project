library(testthat)
library(yaml)
library(purrr) # For list manipulation, if needed

# Helper function to get fixture path
fixture_path <- function(...) {
    testthat::test_path("fixtures", ...)
}

context("YAML Configuration File Reading")

test_that("Can read a simple default config file directly", {
    default_file <- fixture_path("sample_default.yml")
    expect_true(file.exists(default_file), "Fixture file 'sample_default.yml' should exist.")

    config_content <- NULL
    expect_no_error(config_content <- yaml::read_yaml(default_file))

    # Basic structure checks
    expect_type(config_content, "list")
    expect_named(config_content, c("data", "model", "paths"))
    expect_named(config_content$data, c("filename", "target_variable", "features_to_drop"))
    expect_named(config_content$model, c("seed", "cv_folds"))

    # Value checks
    expect_equal(config_content$data$target_variable, "TestTarget")
    expect_equal(config_content$model$seed, 3)
    expect_equal(config_content$paths$artifacts, "test_artifacts")
    expect_length(config_content$data$features_to_drop, 2)
    expect_equal(config_content$data$features_to_drop[[1]], "ID")
})

test_that("Can read the main config file directly", {
    config_file <- fixture_path("sample_config.yml")
    expect_true(file.exists(config_file), "Fixture file 'sample_config.yml' should exist.")

    config_content <- NULL
    expect_no_error(config_content <- yaml::read_yaml(config_file))

    # Basic structure checks
    expect_type(config_content, "list")
    expect_named(config_content, c("default", "testing"))
    expect_named(config_content$default, c("inherits", "report_name"))
    expect_named(config_content$testing, c("inherits", "model", "paths", "report_name"))
    expect_named(config_content$testing$model, "cv_folds")
    expect_named(config_content$testing$paths, "logs")


    # Value checks
    expect_equal(config_content$default$inherits, "sample_default.yml")
    expect_equal(config_content$default$report_name, "default_report")
    expect_equal(config_content$testing$inherits, "sample_default.yml")
    expect_equal(config_content$testing$model$cv_folds, 2)
    expect_equal(config_content$testing$paths$logs, "test_logs")
    expect_equal(config_content$testing$report_name, "testing_report")
})

# Optional: Test manual merging simulation
test_that("Can manually simulate config merging (default)", {
    default_content <- yaml::read_yaml(fixture_path("sample_default.yml"))
    config_content <- yaml::read_yaml(fixture_path("sample_config.yml"))

    # Simulate 'default' environment merge
    # In this case, 'default' only adds 'report_name' and inherits 'sample_default.yml'
    # A simple merge might just take the default_content and add the unique fields from config$default
    merged_default <- default_content
    merged_default$report_name <- config_content$default$report_name # Add the specific field

    # Check the manually merged structure
    expect_equal(merged_default$data$target_variable, "TestTarget")
    expect_equal(merged_default$model$cv_folds, 7)
    expect_equal(merged_default$report_name, "default_report")
})

test_that("Can manually simulate config merging (testing)", {
    default_content <- yaml::read_yaml(fixture_path("sample_default.yml"))
    config_content <- yaml::read_yaml(fixture_path("sample_config.yml"))

    # Simulate 'testing' environment merge
    # Start with default, then override/add from config$testing
    # Use purrr::list_modify for recursive merging (or base R equivalent)
    merged_testing <- purrr::list_modify(default_content, !!!config_content$testing)
    # Note: list_modify only overwrites top-level. Recursive needed for nested like 'model'.
    # A more robust manual merge:
    merged_testing_better <- default_content
    merged_testing_better$model <- purrr::list_modify(default_content$model, !!!config_content$testing$model)
    merged_testing_better$paths <- purrr::list_modify(default_content$paths, !!!config_content$testing$paths)
    merged_testing_better$report_name <- config_content$testing$report_name


    # Check the manually merged structure
    expect_equal(merged_testing_better$data$target_variable, "TestTarget")
    expect_equal(merged_testing_better$model$cv_folds, 2)
    expect_equal(merged_testing_better$model$seed, 3)
    expect_equal(merged_testing_better$paths$artifacts, "test_artifacts")
    expect_equal(merged_testing_better$paths$logs, "test_logs")
    expect_equal(merged_testing_better$report_name, "testing_report")
})
