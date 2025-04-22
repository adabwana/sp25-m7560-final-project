# Load required libraries
library(here)
library(readr)
library(lubridate)
library(dplyr)
library(ggplot2)

library(tseries) # For time series analysis (ACF, Ljung-Box, ADF) adf.test
library(lmtest) # For Durbin-Watson test
library(patchwork) # For combining plots
library(skimr) # For summary statistics

theme_set(theme_bw())

# -----------------------------------------------------------------------------
# READ DATA
# -----------------------------------------------------------------------------
# here() starting path is root of the project
data_raw <- readr::read_csv(here("data", "raw", "LC_train.csv"))
data_eng <- readr::read_csv(here("data", "processed", "train_engineered.csv"))

lc_data <- data_raw %>%
  # Convert dates and times to appropriate formats
  mutate(
    Check_In_Date = mdy(Check_In_Date),
    Check_In_Time = hms::as_hms(Check_In_Time),
    Check_Out_Time = hms::as_hms(Check_Out_Time)
  ) %>%
  # Sort in ascending order
  arrange(Check_In_Date, Check_In_Time) %>%
  # Group by each date
  group_by(Check_In_Date) %>%
  mutate(
    # Cumulative check-ins
    Cum_Arrivals = row_number(), # - 1,  MINUS ONE TO START AT 0 OCCUPANCY AS 1st PERSON ARRIVES
    # Cumulative check-outs
    Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
      sum(!is.na(Check_Out_Time[1:i]) &
        Check_Out_Time[1:i] <= Check_In_Time[i])
    }),
    # Current occupancy
    Occupancy = Cum_Arrivals - Cum_Departures,
    # Course_Code_by_Thousands = as.factor(Course_Code_by_Thousands)
  ) %>%
  ungroup() # %>%
# Remove intermediate columns
# select(-c(Check_Out_Time, Cum_Arrivals, Cum_Departures))

# Basic overview of the data
glimpse(lc_data)
glimpse(data_eng)

# Check for missing values
missing_values <- colSums(is.na(lc_data))
print("Missing values by column:")
print(missing_values[missing_values > 0])

# Basic visualizations


# =================================================================================
# PART A: DURATION IN MINUTES SEQUENTIAL DEPENDENCIES
# =================================================================================

# Combine date and time for ordering and plotting
# Ensure lc_data is sorted by Check_In_Date and Check_In_Time already from lines 23-24
lc_data_ordered <- lc_data %>%
  mutate(Check_In_DateTime = as.POSIXct(paste(Check_In_Date, Check_In_Time), format = "%Y-%m-%d %H:%M:%S")) %>%
  # Arrange again just to be absolutely sure, especially if lc_data was modified
  arrange(Check_In_DateTime)

# 1. Visual Inspection: Plot Duration over Time
g_duration_seq <- ggplot(lc_data_ordered, aes(x = Check_In_DateTime, y = Duration_In_Min)) +
  geom_line(alpha = 0.5, color = "steelblue") +
  # geom_point(alpha = 0.1, size = 0.5) + # Optional: Add points if needed
  labs(
    title = "Sequence Plot of Session Durations",
    x = "Check-In Time",
    y = "Duration (Minutes)"
  ) +
  theme_minimal()

print(g_duration_seq)

# Save the plot
# ggsave(here("presentation", "images", "eda", "duration_sequence.jpg"), g_duration_seq, width = 12, height = 6, dpi = 300)


# 2. Autocorrelation Analysis
# Create a time series object - ensure no NA values if required by functions
# Handle potential NAs in Duration_In_Min if any exist, e.g., na.omit or imputation
duration_ts <- ts(lc_data_ordered$Duration_In_Min) # Assumes lc_data_ordered has no NA duration

# ACF Plot
# Suppress plot output from acf/pacf directly to potentially combine later
acf_result <- acf(duration_ts, main = "ACF for Session Durations", lag.max = 50, plot = FALSE)
pacf_result <- pacf(duration_ts, main = "PACF for Session Durations", lag.max = 50, plot = FALSE)

# Plot ACF and PACF using ggplot for better aesthetics if desired or just use base R plots
# Base R plots:
par(mfrow = c(1, 2)) # Arrange plots side-by-side
plot(acf_result)
plot(pacf_result)
par(mfrow = c(1, 1)) # Reset plotting layout

# Portmanteau Tests (Ljung-Box)
# Test for autocorrelation up to various lags
ljung_box_test_lag10 <- Box.test(duration_ts, lag = 10, type = "Ljung-Box")
ljung_box_test_lag20 <- Box.test(duration_ts, lag = 20, type = "Ljung-Box")
ljung_box_test_lag50 <- Box.test(duration_ts, lag = 50, type = "Ljung-Box")
# Box.test(duration_ts, lag = 100, type = "Ljung-Box")

print("--- Ljung-Box Test Results (Duration) ---")
print(paste("Lag 10:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag10$statistic, ljung_box_test_lag10$parameter, ljung_box_test_lag10$p.value)))
print(paste("Lag 20:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag20$statistic, ljung_box_test_lag20$parameter, ljung_box_test_lag20$p.value)))
print(paste("Lag 50:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag50$statistic, ljung_box_test_lag50$parameter, ljung_box_test_lag50$p.value)))
# Interpretation: Small p-values reject the null hypothesis of independence, suggesting serial correlation.


# 3. Grouping by Attributes - Placeholder/Future Work
# Example: Check ACF within groups (e.g., by Student_IDs)
# grouped_acf <- lc_data_ordered %>%
#   group_by(Student_IDs) %>%
#   summarise(acf_results = list(acf(Duration_In_Min, plot=FALSE)), .groups = 'drop')
# This can be complex to interpret and visualize collectively. Start with overall trend.

# plot(grouped_acf$acf_results[[7]])

# 4. Formal Statistical Tests

# Durbin-Watson Test: Assess autocorrelation in residuals of a model.
# Let's fit a simple model: Duration ~ linear time trend
# Use row number as a proxy for time index if Check_In_DateTime isn't suitable directly
lc_data_ordered <- lc_data_ordered %>% mutate(time_index = row_number())
simple_model <- lm(Duration_In_Min ~ time_index, data = lc_data_ordered)
dw_test_result <- dwtest(simple_model)
print("--- Durbin-Watson Test Results (Duration ~ Time Index) ---")
print(dw_test_result)
# Interpretation: Value close to 2 suggests no first-order autocorrelation in residuals.
# Values < 2 suggest positive autocorrelation, > 2 suggest negative autocorrelation.

# Augmented Dickey-Fuller Test: Check for stationarity.
adf_test_result <- adf.test(duration_ts, alternative = "stationary")
print("--- Augmented Dickey-Fuller Test Results (Duration) ---")
print(adf_test_result)
# Interpretation: Small p-value suggests rejecting the null hypothesis (non-stationarity),
# indicating the time series is likely stationary.


# 5. Domain Knowledge and Experimentation - Interpretation
# Based on ACF, Ljung-Box, DW, and ADF tests, interpret the findings.
# Significant autocorrelation (ACF spikes, small Ljung-Box p-value) suggests temporal dependence.
# Stationarity (small ADF p-value) doesn't rule out autocorrelation.

# --- END OF DURATION SEQUENTIAL DEPENDENCY ANALYSIS ---


# =================================================================================
# PART B: OCCUPANCY DISTRIBUTION ANALYSIS
# =================================================================================

# 1. Visual Inspection: Plot Duration over Time
g_occupancy_seq <- ggplot(lc_data_ordered, aes(x = Check_In_DateTime, y = Occupancy)) +
  geom_line(alpha = 0.5, color = "steelblue") +
  # geom_point(alpha = 0.1, size = 0.5) + # Optional: Add points if needed
  labs(
    title = "Sequence Plot of Session Durations",
    x = "Check-In Time",
    y = "Occupancy"
  ) +
  theme_minimal()

print(g_occupancy_seq)

# Save the plot
# ggsave(here("presentation", "images", "eda", "duration_sequence.jpg"), g_duration_seq, width = 12, height = 6, dpi = 300)


# 2. Autocorrelation Analysis
# Create a time series object - ensure no NA values if required by functions
# Handle potential NAs in Duration_In_Min if any exist, e.g., na.omit or imputation
occupancy_ts <- ts(lc_data_ordered$Occupancy) # Assumes lc_data_ordered has no NA duration

# ACF Plot
# Suppress plot output from acf/pacf directly to potentially combine later
acf_result <- acf(occupancy_ts, main = "ACF for Session Durations", lag.max = 50, plot = FALSE)
pacf_result <- pacf(occupancy_ts, main = "PACF for Session Durations", lag.max = 50, plot = FALSE)

# Plot ACF and PACF using ggplot for better aesthetics if desired or just use base R plots
# Base R plots:
par(mfrow = c(1, 2)) # Arrange plots side-by-side
plot(acf_result)
plot(pacf_result)
par(mfrow = c(1, 1)) # Reset plotting layout

# Portmanteau Tests (Ljung-Box)
# Test for autocorrelation up to various lags
ljung_box_test_lag10 <- Box.test(occupancy_ts, lag = 10, type = "Ljung-Box")
ljung_box_test_lag20 <- Box.test(occupancy_ts, lag = 20, type = "Ljung-Box")
ljung_box_test_lag50 <- Box.test(occupancy_ts, lag = 50, type = "Ljung-Box")
# Box.test(duration_ts, lag = 100, type = "Ljung-Box")

print("--- Ljung-Box Test Results (Occupancy) ---")
print(paste("Lag 10:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag10$statistic, ljung_box_test_lag10$parameter, ljung_box_test_lag10$p.value)))
print(paste("Lag 20:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag20$statistic, ljung_box_test_lag20$parameter, ljung_box_test_lag20$p.value)))
print(paste("Lag 50:", sprintf("X-squared = %.3f, df = %d, p-value = %.3g", ljung_box_test_lag50$statistic, ljung_box_test_lag50$parameter, ljung_box_test_lag50$p.value)))
# Interpretation: Small p-values reject the null hypothesis of independence, suggesting serial correlation.


# 3. Grouping by Attributes - Placeholder/Future Work
# Example: Check ACF within groups (e.g., by Student_IDs)
# grouped_acf <- lc_data_ordered %>%
#   group_by(Student_IDs) %>%
#   summarise(acf_results = list(acf(Duration_In_Min, plot=FALSE)), .groups = 'drop')
# This can be complex to interpret and visualize collectively. Start with overall trend.

# plot(grouped_acf$acf_results[[7]])

# 4. Formal Statistical Tests

# Durbin-Watson Test: Assess autocorrelation in residuals of a model.
# Let's fit a simple model: Duration ~ linear time trend
# Use row number as a proxy for time index if Check_In_DateTime isn't suitable directly
lc_data_ordered <- lc_data_ordered %>% mutate(time_index = row_number())
simple_model <- lm(Duration_In_Min ~ time_index, data = lc_data_ordered)
dw_test_result <- dwtest(simple_model)
print("--- Durbin-Watson Test Results (Occupancy ~ Time Index) ---")
print(dw_test_result)
# Interpretation: Value close to 2 suggests no first-order autocorrelation in residuals.
# Values < 2 suggest positive autocorrelation, > 2 suggest negative autocorrelation.

# Augmented Dickey-Fuller Test: Check for stationarity.
adf_test_result <- adf.test(occupancy_ts, alternative = "stationary")
print("--- Augmented Dickey-Fuller Test Results (Occupancy) ---")
print(adf_test_result)
# Interpretation: Small p-value suggests rejecting the null hypothesis (non-stationarity),
# indicating the time series is likely stationary.


# 5. Domain Knowledge and Experimentation - Interpretation
# Based on ACF, Ljung-Box, DW, and ADF tests, interpret the findings.
# Significant autocorrelation (ACF spikes, small Ljung-Box p-value) suggests temporal dependence.
# Stationarity (small ADF p-value) doesn't rule out autocorrelation.


# =================================================================================
# PART C: CORRELATION ANALYSIS
# =================================================================================
raw_colnames <- colnames(data_raw)

data_pairs_numeric <- data_eng %>%
  dplyr::select(all_of(raw_colnames), Session_Length_Category, Occupancy) %>%
  dplyr::select(where(is.numeric), Session_Length_Category) %>%
  # Calculate quartiles and upper fence for Duration_In_Min
  dplyr::mutate(
    Q1 = quantile(Duration_In_Min, 0.25),
    Q2 = quantile(Duration_In_Min, 0.5),
    Q3 = quantile(Duration_In_Min, 0.75),
    IQR = Q3 - Q2,
    Upper_Fence = Q3 + 1.5 * IQR,
    Way_Out = Q3 + 3 * IQR,
    Fugedaboudit = Q3 + 7 * IQR,
    Session_Length_Category = case_when(
      Duration_In_Min <= Q3 ~ "Short",
      Duration_In_Min <= Way_Out ~ "Medium",
      Duration_In_Min <= Fugedaboudit ~ "Long",
      TRUE ~ "Extended"
    )
  ) %>%
  # Remove the helper columns
  dplyr::select(-Q1, -Q2, -Q3, -IQR, -Upper_Fence, -Way_Out, -Fugedaboudit) %>%
  # Reorder final columns
  dplyr::select(
    -Duration_In_Min, -Occupancy, -Session_Length_Category,
    Duration_In_Min, Occupancy, Session_Length_Category
  )

# Let's check the distribution
print("Distribution of Session Length Categories:")
print(table(data_pairs_numeric$Session_Length_Category))

# And verify some statistics
print("Summary of Duration by Category:")
print(tapply(
  data_pairs_numeric$Duration_In_Min,
  data_pairs_numeric$Session_Length_Category,
  summary
))

quantile(data_pairs_numeric$Duration_In_Min, 0.25)
quantile(data_pairs_numeric$Duration_In_Min, 0.75)

data_pairs_categorical <- data_eng %>%
  dplyr::select(all_of(raw_colnames), Session_Length_Category, Occupancy) %>%
  dplyr::select(where(is.character), Duration_In_Min, Occupancy) %>%
  # Calculate quartiles and upper fence for Duration_In_Min
  dplyr::mutate(
    Q1 = quantile(Duration_In_Min, 0.25),
    Q2 = quantile(Duration_In_Min, 0.5),
    Q3 = quantile(Duration_In_Min, 0.75),
    IQR = Q3 - Q2,
    Upper_Fence = Q3 + 1.5 * IQR,
    Way_Out = Q3 + 3 * IQR,
    Fugedaboudit = Q3 + 7 * IQR,
    Session_Length_Category = case_when(
      Duration_In_Min <= Q3 ~ "Short",
      Duration_In_Min <= Way_Out ~ "Medium",
      Duration_In_Min <= Fugedaboudit ~ "Long",
      TRUE ~ "Extended"
    )
  ) %>%
  # Remove the helper columns
  dplyr::select(-Q1, -Q2, -Q3, -IQR, -Upper_Fence, -Way_Out, -Fugedaboudit) %>%
  # Reorder final columns
  dplyr::select(
    -Duration_In_Min, -Occupancy, -Session_Length_Category,
    Duration_In_Min, Occupancy, Session_Length_Category
  )

data_pairs_categorical2 <- data_pairs_categorical %>%
  dplyr::select(where(~ n_distinct(.) <= 10), Duration_In_Min, Occupancy)

# Verify the results
data_pairs_categorical %>%
  summarise(across(everything(), n_distinct)) %>%
  glimpse()
