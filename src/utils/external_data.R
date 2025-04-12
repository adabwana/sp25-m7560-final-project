library(here)
library(readr)
library(dplyr)
library(lubridate)
library(openmeteo)
library(lunar)
library(hms) # Required for hms::as_hms

# Read the engineered data
# Using show_col_types = FALSE to suppress messages in console
train_engineered <- readr::read_csv(here("data", "processed", "train_engineered.csv"), show_col_types = FALSE)
test_engineered <- readr::read_csv(here("data", "processed", "test_engineered.csv"), show_col_types = FALSE)

# Add origin identifier and combine
train_engineered <- train_engineered %>% mutate(origin = "train")
test_engineered <- test_engineered %>% mutate(origin = "test")
data_full <- bind_rows(train_engineered, test_engineered)

# Ensure correct data types for date/time columns from engineered data
data_full <- data_full %>%
  mutate(
    Check_In_Date = as.Date(Check_In_Date), 
    Check_In_Time = hms::as_hms(Check_In_Time) 
  )

# -----------------------------------------------------------------------------
# MOON PHASES
# -----------------------------------------------------------------------------
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
# Get the start and end dates from the combined data
start_date <- min(data_full$Check_In_Date)
end_date <- max(data_full$Check_In_Date)

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
# (as specified in the weather_history call in the original file)
all_hourly_vars <- weather_variables()[["hourly_history_vars"]]

# Select the hourly weather variables to get
hourly_vars <- c("cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high", 
  "temperature_2m", "windspeed_10m", "precipitation", "rain", "snowfall")

# Get the weather history
weather_history <- weather_history(
  location = bowling_green_coords,
  start = start_date,
  end = end_date,
  response_units = response_units,
  hourly = all_hourly_vars
)

# Prepare weather data for joining
weather_to_join <- weather_history %>%
  select(datetime, starts_with("hourly_")) %>% 
  # Add 'weather_' prefix to all weather variables for clarity and to avoid name conflicts
  rename_with(~ paste0("weather_", .), starts_with("hourly_")) %>%
  rename(Hourly_Timestamp_Floor = datetime)

# --- Merge Weather Data ---
# Create Check_In_Datetime and the hourly floor timestamp in the main data
data_full <- data_full %>%
  mutate(
    Check_In_Datetime = ymd_hms(paste(Check_In_Date, Check_In_Time), quiet = TRUE), # Combine Date and Time
    Hourly_Timestamp_Floor = floor_date(Check_In_Datetime, "hour") # Floor to the hour
  )

# Join weather data to the main dataframe
data_full_ext <- left_join(data_full, weather_to_join, by = "Hourly_Timestamp_Floor")

# Remove intermediate datetime columns used for joining
data_full_ext <- data_full_ext %>%
  select(-Check_In_Datetime, -Hourly_Timestamp_Floor)

# -----------------------------------------------------------------------------
# SPLIT AND SAVE RESULTS
# -----------------------------------------------------------------------------
# Split back into train and test based on the 'origin' column
train_eng_ext <- data_full_ext %>% filter(origin == "train") %>% select(-origin)
test_eng_ext <- data_full_ext %>% filter(origin == "test") %>% select(-origin)

# Ensure the processed directory exists
dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)

# Save the results
readr::write_csv(train_eng_ext, here("data", "processed", "train_eng_ext.csv"))
readr::write_csv(test_eng_ext, here("data", "processed", "test_eng_ext.csv"))

print("External data (moon phase and weather) added and saved to train_eng_ext.csv and test_eng_ext.csv")