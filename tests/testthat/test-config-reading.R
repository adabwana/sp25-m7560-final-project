library(testthat)
library(yaml) # Use yaml directly in tests if needed, but primarily testing load_config
# library(config) # No longer needed for load_config tests

# Source the utility functions to be tested
source(here::here("src/r/utils/config_utils.R"))

# Define absolute path to the fixture file
fixture_file <- here::here("tests", "testthat", "fixtures", "config.yml")

context("Configuration Loading (yaml::read_yaml wrapper) and Value Retrieval")

# --- Tests for load_config() ---

test_that("load_config() successfully reads the fixture config.yml via absolute path", {
    cfg <- NULL
    suppressMessages({
        expect_no_error(cfg <- load_config(file = fixture_file))
    })

    expect_type(cfg, "list")
    # Fixture should NOT have top-level default key
    expect_named(cfg, c("data", "model", "parallel", "paths", "logging"))
    expect_equal(cfg$data$filename, "test_data.csv")
    expect_equal(cfg$model$seed, 42)
    expect_false(cfg$parallel$enabled)
})

test_that("load_config() handles relative path from project root", {
    # Assumes test is run from project root
    relative_path <- "tests/testthat/fixtures/config.yml"
    cfg <- NULL
    suppressMessages({
        expect_no_error(cfg <- load_config(file = relative_path))
    })
    expect_type(cfg, "list")
    expect_equal(cfg$data$filename, "test_data.csv")
})

test_that("load_config() uses default path if none provided", {
    # This test requires the actual config/config.yml to exist and be readable
    # It might be better to mock this or skip if main config isn't guaranteed
    # For now, let's assume it exists for a basic check
    skip_if_not(file.exists(here::here("config", "config.yml")), "Main config/config.yml not found")
    cfg <- NULL
    suppressMessages({
        expect_no_error(cfg <- load_config()) # Use default file path
    })
    expect_type(cfg, "list")
    # Add a basic check if possible, e.g., expect_true("data" %in% names(cfg))
    expect_true(length(cfg) > 0)
})

test_that("load_config() errors correctly with non-existent path", {
    non_existent_path <- here::here("tests", "testthat", "fixtures", "non_existent.yml")
    relative_non_existent <- "tests/testthat/fixtures/non_existent.yml"

    suppressMessages({
        expect_error(
            load_config(file = non_existent_path),
            "Configuration file path does not exist"
        )
        expect_error(
            load_config(file = relative_non_existent),
            "Configuration file path does not exist"
        )
        expect_error(
            load_config(file = "completely_made_up_file.yml"),
            "Configuration file path does not exist"
        )
    })
})

# --- Tests for get_config_value() ---

# Load config once for these tests using the absolute path
config_fixture <- suppressMessages(load_config(file = fixture_file))

# Ensure config_fixture loaded correctly before running dependent tests
stopifnot(!is.null(config_fixture) && length(config_fixture) > 0)

test_that("get_config_value() retrieves top-level values", {
    expect_equal(get_config_value(config_fixture, "model"), config_fixture$model)
    expect_type(get_config_value(config_fixture, "model"), "list")
    expect_null(get_config_value(config_fixture, "non_existent_key"))
})

test_that("get_config_value() retrieves nested values using dot notation", {
    expect_equal(get_config_value(config_fixture, "data.filename"), "test_data.csv")
    expect_equal(get_config_value(config_fixture, "parallel.enabled"), FALSE)
    expect_equal(get_config_value(config_fixture, "paths.models"), "test_artifacts/models")
    expect_null(get_config_value(config_fixture, "data.non_existent_sub_key"))
    expect_null(get_config_value(config_fixture, "model.non_existent.deeper"))
})

test_that("get_config_value() returns default value when path not found", {
    expect_equal(get_config_value(config_fixture, "non_existent_key", default = "default_val"), "default_val")
    expect_equal(get_config_value(config_fixture, "data.non_existent_sub_key", default = 123), 123)
    expect_false(get_config_value(config_fixture, "parallel.something_else", default = FALSE))

    # Check that it returns actual value if it exists, not default
    expect_equal(get_config_value(config_fixture, "model.seed", default = 999), 42)
})

test_that("get_config_value() returns default value when retrieved value is NULL", {
    # Add a NULL value to the fixture for testing this
    config_with_null <- config_fixture
    config_with_null$logging$optional_setting <- NULL

    expect_null(get_config_value(config_with_null, "logging.optional_setting"))
    expect_equal(get_config_value(config_with_null, "logging.optional_setting", default = "was_null"), "was_null")

    # Ensure non-null values are not replaced by default
    expect_equal(get_config_value(config_with_null, "logging.level", default = "was_null"), "DEBUG")
})
