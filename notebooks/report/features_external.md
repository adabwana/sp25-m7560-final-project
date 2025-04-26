# External Data

The complete implementation for integrating external data can be found in our [source code](https://github.com/adabwana/sp25-m7560-final-project/blob/master/src/r/utils/create_data/external_data.R).

Beyond the features derived directly from the Learning Commons dataset, we incorporated external data sources to potentially capture broader environmental influences on student behavior.

## Data Handling

Before adding external features, the script first combines the `train_engineered.csv` and `test_engineered.csv` files (output from the primary feature engineering script). An `origin` column is added to track which dataset each row came from. This allows external data fetching (like weather) to be done efficiently across the entire date range. After adding external features, the combined dataset is split back into `train` and `test` sets based on the `origin` column before saving the final outputs.

## Moon Phase Features

We (Jaryt) hypothesized (wanted to Jason to cringe) that lunar cycles might subtly influence activity patterns. To explore this, we used the R `lunar` package to calculate the moon phase for each `Check_In_Date`.

```r
# Relevant snippet from src/r/utils/create_data/external_data.R
library(lunar)

data_full <- data_full %>%
  mutate(
    Moon_Phase_Radians = lunar::lunar.phase(Check_In_Date, name = FALSE), # Get numeric phase in radians
    Cosine_Phase = cos(Moon_Phase_Radians)
    # Original code also calculated 8 discrete phases, but only Cosine_Phase was kept
  ) %>%
  select(-Moon_Phase_Radians) # Remove the intermediate radians column
```

The `lunar.phase` function returns the phase as a value in radians. We then calculated the cosine of this value (`Cosine_Phase`). Using the cosine transforms the cyclical phase information into a continuous variable ranging from -1 to 1, which can be more easily incorporated into regression models than categorical phase names.

## Weather Features

Local weather conditions could plausibly affect whether students choose to visit the Learning Commons or how long they stay. We used the `openmeteo` R package to fetch historical hourly weather data for Bowling Green, Ohio.

1.  **Define Parameters**: We identified the date range from our combined dataset and specified the geographic coordinates for Bowling Green.

    ```r
    # Relevant snippet from src/r/utils/create_data/external_data.R
    start_date <- min(data_full$Check_In_Date)
    end_date <- max(data_full$Check_In_Date)

    bowling_green_coords <- c(
      latitude = 41.374775,
      longitude = -83.651321
    )
    ```

2.  **Fetch Data**: We requested various hourly metrics (including temperature, cloud cover, wind speed, precipitation, rain, snowfall, etc.) using the `weather_history` function. Error handling was included in case the API request failed.

    ```r
    library(openmeteo)
    
    response_units <- list(
      temperature_unit = "fahrenheit",
      windspeed_unit = "kmh",
      precipitation_unit = "mm"
    )
    
    all_hourly_vars <- weather_variables()[["hourly_history_vars"]]
    
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
    ```

3.  **Prepare and Merge**: If the weather data was fetched successfully, we prepared it for joining. This involved selecting relevant columns, renaming them with a `weather_` prefix, and creating a common `Hourly_Timestamp_Floor` key in both the weather data and our main dataset (by flooring the `Check_In_Datetime` to the beginning of the hour as preventing future weather from influencing a student's decision to come or go to the LC). A `left_join` merged the corresponding hourly weather data onto each visit record.

    ```r
    # Relevant snippet from src/r/utils/create_data/external_data.R
    if (!is.null(weather_history)) {
        weather_to_join <- weather_history %>%
          select(datetime, starts_with("hourly_")) %>%
          rename_with(~ paste0("weather_", .), starts_with("hourly_")) %>%
          rename(Hourly_Timestamp_Floor = datetime)
          
        data_full <- data_full %>%
          mutate(
            Check_In_Datetime = ymd_hms(paste(Check_In_Date, Check_In_Time), quiet = TRUE),
            Hourly_Timestamp_Floor = floor_date(Check_In_Datetime, "hour")
          )
          
        data_full_ext <- left_join(data_full, weather_to_join, by = "Hourly_Timestamp_Floor")
        
        data_full_ext <- data_full_ext %>%
          select(-Check_In_Datetime, -Hourly_Timestamp_Floor)
    } else {
        # Handle case where weather fetch failed
        data_full_ext <- data_full
    }
    ```

## Final Output

After potentially adding the moon phase and weather features, the combined dataset was split back into training and testing sets based on the original `origin` flag and saved as `train_engineered.csv` and `test_engineered.csv` in the `data/processed` directory, overwriting the intermediate files generated by `feature_engineering.R`.