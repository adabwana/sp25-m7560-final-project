library(here)
library(readr)
library(dplyr)
library(lubridate)
library(openmeteo)
library(lunar)
library(hms) # Required for hms::as_hms
library(glue) # Add glue

# Define filenames (to avoid reusing hardcoded strings)
INPUT_TRAIN_FILE <- here("data", "processed", "xxxtrain_engineered.csv")
INPUT_TEST_FILE <- here("data", "processed", "xxxtest_engineered.csv")
OUTPUT_TRAIN_FILE <- here("data", "processed", "train_engineered.csv")
OUTPUT_TEST_FILE <- here("data", "processed", "test_engineered.csv")

# Check if input files exist
if (!file.exists(INPUT_TRAIN_FILE) || !file.exists(INPUT_TEST_FILE)) {
  stop(glue::glue("Input files not found. Please run feature_engineering.R first. Expected: \n{INPUT_TRAIN_FILE}\n{INPUT_TEST_FILE}"))
}

cat(glue::glue("\n--- Loading initial engineered data ---\n"))
train_engineered <- readr::read_csv(INPUT_TRAIN_FILE, show_col_types = FALSE)
test_engineered <- readr::read_csv(INPUT_TEST_FILE, show_col_types = FALSE)

# Add origin identifier and combine
cat("--- Combining train and test data ---\n")
train_engineered <- train_engineered %>% mutate(origin = "train")
test_engineered <- test_engineered %>% mutate(origin = "test")
data_full <- bind_rows(train_engineered, test_engineered)

# Ensure correct data types 
cat("--- Ensuring correct date/time types ---\n")
data_full <- data_full %>%
  mutate(
    Check_In_Date = as.Date(Check_In_Date), 
    Check_In_Time = hms::as_hms(Check_In_Time) 
  )

# -----------------------------------------------------------------------------
# MOON PHASES
# -----------------------------------------------------------------------------
cat("--- Calculating moon phases ---\n\n")
# Calculate moon phases (only keep Cosine_Phase as per previous steps)
data_full <- data_full %>%
  mutate(
    Moon_Phase_Radians = lunar::lunar.phase(Check_In_Date, name = FALSE), # Get numeric phase in radians
    Cosine_Phase = cos(Moon_Phase_Radians),
    # Moon_8Phases = case_when(
    #   Moon_Phase <= pi/8 | Moon_Phase > 15*pi/8 ~ "New",
    #   Moon_Phase > pi/8 & Moon_Phase <= 3*pi/8 ~ "Waxing crescent",
    #   Moon_Phase > 3*pi/8 & Moon_Phase <= 5*pi/8 ~ "First quarter",
    #   Moon_Phase > 5*pi/8 & Moon_Phase <= 7*pi/8 ~ "Waxing gibbous",
    #   Moon_Phase > 7*pi/8 & Moon_Phase <= 9*pi/8 ~ "Full",
    #   Moon_Phase > 9*pi/8 & Moon_Phase <= 11*pi/8 ~ "Waning gibbous",
    #   Moon_Phase > 11*pi/8 & Moon_Phase <= 13*pi/8 ~ "Last quarter",
    #   Moon_Phase > 13*pi/8 & Moon_Phase <= 15*pi/8 ~ "Waning crescent"
    # ),

  ) #%>%
  # select(-Moon_Phase_Radians) # Remove the intermediate radians column

# -----------------------------------------------------------------------------
# WEATHER HISTORY
# -----------------------------------------------------------------------------
cat("--- Fetching historical weather data from OpenMeteo ---\n\n")
# Get the start and end dates from the combined data
start_date <- min(data_full$Check_In_Date)
end_date <- max(data_full$Check_In_Date)
cat(glue("Weather date range: {start_date} to {end_date}\n\n"))

# Bowling Green coordinates
bowling_green_coords <- c(
  latitude = 41.374775,
  longitude = -83.651321
)

# OpenMeteo metrics (using units consistent with previous requests)
response_units <- list(
  temperature_unit = "fahrenheit",
  windspeed_unit = "kmh",
  precipitation_unit = "mm"
)

# Select all available hourly weather variables 
all_hourly_vars <- weather_variables()[["hourly_history_vars"]]

# Select the hourly weather variables to get
hourly_vars <- c("cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high", 
  "temperature_2m", "windspeed_10m", "precipitation", "rain", "snowfall")

# Get the weather history (this can take a moment)
weather_history <- tryCatch({
    weather_history(
      location = bowling_green_coords,
      start = start_date,
      end = end_date,
      response_units = response_units,
      hourly = all_hourly_vars
    )
}, error = function(e) {
    warning(glue("Failed to fetch weather data: {e$message}\nProceeding without weather features."), call. = FALSE)
    NULL # Return NULL if fetching fails
})

# --- Merge Weather Data (only if fetched successfully) ---
if (!is.null(weather_history)) {
    cat("--- Preparing and merging weather data ---\n\n")
    weather_to_join <- weather_history %>%
      select(datetime, starts_with("hourly_")) %>% 
      rename_with(~ paste0("weather_", .), starts_with("hourly_")) %>%
      rename(Hourly_Timestamp_Floor = datetime)
      
    # Create Check_In_Datetime and the hourly floor timestamp in the main data
    data_full <- data_full %>%
      mutate(
        Check_In_Datetime = ymd_hms(paste(Check_In_Date, Check_In_Time), quiet = TRUE), 
        Hourly_Timestamp_Floor = floor_date(Check_In_Datetime, "hour") 
      )
      
    # Join weather data to the main dataframe
    data_full_ext <- left_join(data_full, weather_to_join, by = "Hourly_Timestamp_Floor")
    
    # Remove intermediate datetime columns used for joining
    data_full_ext <- data_full_ext %>%
      select(-Check_In_Datetime, -Hourly_Timestamp_Floor)
} else {
    cat("--- Skipping weather data merge due to fetch failure ---\n\n")
    # If weather failed, use the data_full without weather columns
    data_full_ext <- data_full # Assign data_full so the rest of the script works
}


# -----------------------------------------------------------------------------
# SPLIT AND SAVE RESULTS
# -----------------------------------------------------------------------------
cat("--- Splitting data back into train and test sets ---\n\n")
# Split back into train and test based on the 'origin' column
train_eng_ext <- data_full_ext %>% filter(origin == "train") %>% select(-origin)
test_eng_ext <- data_full_ext %>% filter(origin == "test") %>% select(-origin)

# Ensure the processed directory exists
dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)

cat(glue::glue("--- Saving final engineered data ---\n\nOutput Train: {OUTPUT_TRAIN_FILE}\n\nOutput Test: {OUTPUT_TEST_FILE}\n\n"))
# Save the results
readr::write_csv(train_eng_ext, OUTPUT_TRAIN_FILE)
readr::write_csv(test_eng_ext, OUTPUT_TEST_FILE)

cat("--- External data processing complete ---\n\n")