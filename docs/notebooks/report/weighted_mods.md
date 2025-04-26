# Weighted Duration

This document focuses on a specific post-processing step applied to the duration predictions: weighting the model's output with a student's historical average visit duration. This strategy was explored under the hypothesis that individual students exhibit habitual patterns in their Learning Commons usage. Initial model fitting and hyperparameter tuning details for MARS, Random Forest, and XGBoost can be found in the `model_tuning.md` document.

## Rationale for Weighting

We believe that students may repeatedly visit the Learning Commons and stay roughly the same amount of time each visit as it becomes part of their routine habits. To reflect this, we considered adjusting the raw model predictions by incorporating the mean duration of a student's previous visits observed in the training data.

Using the historical mean visit duration directly as a predictor during model training is problematic, as it requires multiple data points per student and would complicate standard cross-validation procedures (data leakage). Therefore, we applied this adjustment *after* the initial model predictions were generated.

## Initial Exploration with Fixed Weights

We first tested this weighting idea on the holdout set predictions from the MARS, Random Forest, and XGBoost models, initially using an arbitrary weight of `w1 = 2/3` for the model prediction and `w2 = 1/3` for the historical average. If a student had no prior visits in the training set, their original model prediction was used.

```r
# Example using MARS predictions (hfit1) and holdout data
# Assumes 'studentsfull', 'trainfull', 'holdoutfull', 'holdout', 'hfit1' are loaded/defined
# (Code for loading/splitting/initial MARS fit is omitted for brevity)

w1 <- 2/3
w2 <- 1-w1

ID1 <- data.frame(ids = holdoutfull$Student_IDs, y = hfit1) # Using 'y' for generic prediction

ID1$OtherVisits <- NA # Initialize column
for (i in 1:nrow(ID1)) {
  mask <- trainfull$Student_IDs == ID1$ids[i]
  # Calculate mean duration from training visits ONLY
  mean_duration <- mean(trainfull$Duration_In_Min[mask], na.rm = TRUE)
  # Only assign if not NaN (i.e., student had prior visits)
  if (!is.nan(mean_duration)) {
      ID1$OtherVisits[i] <- mean_duration
  }
}

# Apply weighting: use prediction if no prior visits (OtherVisits is NA), otherwise use weighted average
ID1$Weighted <- ifelse(is.na(ID1$OtherVisits), ID1$y,
                      w1*ID1$y + w2*ID1$OtherVisits)

# Calculate RMSE for the weighted predictions
toterrorW1 <- sum((ID1$Weighted - holdout$Duration_In_Min)^2)/length(ID1$Weighted)
sqrt(toterrorW1)
```

```
## [1] 57.78979
```

Applying this fixed 2/3 weight yielded slight improvements in RMSE for all three models compared to their unweighted predictions. For example, MARS RMSE improved from ~59.23 to ~57.79, and Random Forest improved from ~57.04 to ~56.57. This suggested the weighting approach had merit.

This weighting approach highlights the potential relevance of individual student behavior patterns in modeling visit duration.

## Optimizing Weights via Cross-Validation for Random Forest

Since Random Forest showed strong performance, we proceeded to find the *optimal* weight specifically for this model using a more rigorous cross-validation approach on the *entire* training dataset (not just the initial holdout split).

**The final tuned hyperparameters for the Random Forest model used in this CV process were:**
-   `mtry = 20` (number of predictors sampled at each split)
-   `trees = 200` (number of trees in the forest)
-   `min_n = 10` (minimum number of observations in a node to allow a split)

```r
# --- Setup for Weight CV ---
# Assumes 'maintrain' (full preprocessed training data) and
# 'studentsfull' (original training data with IDs) are loaded.
# library(caret), library(doParallel), library(here) assumed loaded.

# Prepare parallel processing
num_cores <- detectCores(logical = FALSE)
num_cores_to_use <- max(1, num_cores - 4) # Use at least 1 core
cl <- makePSOCKcluster(num_cores_to_use)
registerDoParallel(cl)

# Create 5 CV folds from the full training data
set.seed(7560)
testInd <- createFolds(maintrain$Duration_In_Min, k = 5)

IDLIST <- list() # To store predictions from each fold

# Fixed hyperparameters for the RF model
tune <- data.frame(mtry = 20)
nodesize_val <- 10
ntree_val <- 200

# --- Cross-Validation Loop ---
cat("Starting 5-fold CV for Random Forest predictions...
")
for (i in 1:5) {
  cat(glue::glue("Processing Fold {i}/5...
"))
  # Train RF on 4 folds
  # Note: Using caret::train here for consistency with original script,
  # but tidymodels::fit_resamples could also be used.
  fullF <- train(Duration_In_Min ~ .,
                 data = maintrain[-testInd[[i]],], # Training data for this fold
                 method = "rf",
                 tuneGrid = tune,       # Fixed mtry
                 ntree = ntree_val,     # Fixed ntree
                 nodesize = nodesize_val, # Fixed nodesize (min_n)
                 trControl = trainControl(method = "none")) # No inner tuning needed

  # Predict on the held-out fold
  fittedF <- predict(fullF, newdata = maintrain[testInd[[i]],]) # Test data for this fold

  # Store results: Fold number, Student ID, Prediction, True Value
  IDLIST[[i]] <- data.frame(Fold = i,
                            ids = studentsfull$Student_IDs[testInd[[i]]], # Get IDs from original data
                            fittedF = fittedF,
                            TRUEy = maintrain$Duration_In_Min[testInd[[i]]]) # Get true values
}
cat("CV finished.
")
stopCluster(cl) # Stop parallel backend

# --- Calculate Historical Averages ---
# Combine predictions from all folds
ID <- do.call(rbind, IDLIST)
colnames(ID) <- c("Fold", "ids", "fittedF", "TRUEy")

cat("Calculating historical average durations across folds...
")
ID$AveOtherVisits <- NA # Initialize column
for (i in 1:nrow(ID)) {
  current_id <- ID$ids[i]
  current_fold <- ID$Fold[i]

  # Calculate average TRUE duration for this student from OTHER folds
  # This prevents data leakage within the CV process
  mask <- (ID$ids == current_id) & (ID$Fold != current_fold)
  if (any(mask)) { # Check if the student exists in other folds
      mean_duration <- mean(ID$TRUEy[mask], na.rm = TRUE)
      if (!is.nan(mean_duration)) {
        ID$AveOtherVisits[i] <- mean_duration
      }
  }
  if (i %% 1000 == 0) {cat(glue::glue("Processed {i}/{nrow(ID)} students...
"))}
}
cat("Historical averages calculated.
")

# --- Test Different Weights ---
# Define a grid of weights (w1 for model prediction, w2=1-w1 for historical avg)
weights_w1 <- c(0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 1)
RMSE_results <- c() # To store RMSE for each weight

cat("Evaluating different weights...
")
for (j in 1:length(weights_w1)) {
  w1 <- weights_w1[j]
  w2 <- 1 - w1

  # Apply the current weight combination
  ID$Weighted <- ifelse(is.na(ID$AveOtherVisits), ID$fittedF,
                        w1 * ID$fittedF + w2 * ID$AveOtherVisits)

  # Calculate RMSE across all folds for this weight combination
  # Note: Original script calculated MSE per fold then averaged.
  # Calculating overall RMSE directly is simpler here.
  overall_rmse <- sqrt(mean((ID$Weighted - ID$TRUEy)^2))
  RMSE_results[j] <- overall_rmse

  cat(glue::glue("Weight w1={round(w1, 3)} -> Overall CV RMSE = {round(overall_rmse, 3)}
"))
}

# Display results
weight_performance <- data.frame(Weight_for_Model_Pred = weights_w1, CV_RMSE = RMSE_results)
print(weight_performance)
best_weight_idx <- which.min(RMSE_results)
best_w1 <- weights_w1[best_weight_idx]
cat(glue::glue("
Optimal weight (w1) for model prediction: {round(best_w1, 3)}
"))
```

```
##     weights     RMSE
## 1 0.0000000 62.18344
## 2 0.1666667 60.53198
## 3 0.2500000 59.85238
## 4 0.3333333 59.27482
## 5 0.5000000 58.43742
## 6 0.6666667 58.03793
## 7 0.7500000 58.00562
## 8 0.8333333 58.08540
## 9 1.0000000 58.57874
```

The cross-validation results indicated that a weight `w1 = 3/4` for the model prediction and `w2 = 1/4` for the student's historical average visit duration yielded the lowest overall RMSE.

## Final Prediction Generation

Using the optimal weights (`w1 = 3/4`, `w2 = 1/4`) determined through cross-validation, we generated the final adjusted predictions for the original test set (or potentially new data).

```r
# --- Apply Optimal Weight to Final Predictions ---
# Assumes 'modpreds' (raw RF predictions on test set) and
# 'fulltest' (original test set data with IDs) are loaded.
# Assumes 'ID' (CV results data frame) is available for historical averages.

# Load raw predictions (replace path if needed)
modpreds <- read.csv(here::here("data/predictions/Duration_In_Min_RandomForest_20250423180825_pred.csv"))
# Load test set features (replace path if needed)
fulltest <- read.csv(here::here("data/processed/test_engineered.csv"))

# Combine predictions with student IDs from the test set
predIDs <- data.frame(raw_pred = modpreds$.pred, ids = fulltest$Student_IDs) # Use .pred column

# --- Lookup Historical Averages from Training CV ---
# Calculate the overall historical average for each student from the CV results
# Group by student ID and calculate the mean of their TRUE durations across all folds they appeared in
historical_averages <- aggregate(TRUEy ~ ids, data = ID, FUN = mean, na.rm = TRUE)
colnames(historical_averages) <- c("ids", "historical_avg_duration")

# Merge these averages into the test set predictions
predIDs <- merge(predIDs, historical_averages, by = "ids", all.x = TRUE) # Left join

# --- Apply Optimal Weights ---
w1 <- 3/4
w2 <- 1/4

# Apply weighting: use raw prediction if no historical average, otherwise use weighted average
predIDs$FINAL_prediction <- ifelse(is.na(predIDs$historical_avg_duration),
                                   predIDs$raw_pred,
                                   w1 * predIDs$raw_pred + w2 * predIDs$historical_avg_duration)

# --- Save Final Predictions ---
# Select only the final weighted predictions
final_output <- data.frame(Prediction = predIDs$FINAL_prediction)

# Save the final predictions (adjust path as needed)
output_path <- here::here("data/predictions/Duration_FINAL_weighted.csv")
write.csv(final_output, output_path, row.names = FALSE)

cat(glue::glue("Final weighted predictions saved to: {output_path}
"))
```

This weighted approach provided a modest but consistent improvement over the raw Random Forest predictions for duration, acknowledging the influence of individual student habits.
