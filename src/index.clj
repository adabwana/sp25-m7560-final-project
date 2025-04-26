^:kindly/hide-code
(ns index
  (:require
   [scicloj.kindly.v4.api :as kindly]
   [scicloj.kindly.v4.kind :as kind])
  (:import [java.time LocalDate]
           [java.time.format DateTimeFormatter]))

^:kindly/hide-code
(def md (comp kindly/hide-code kind/md))

(let [formatter (DateTimeFormatter/ofPattern "M/d/yy")
      current-date (str (.format (LocalDate/now) formatter))]
  (md (str "
### Predicting Learning Commons Usage 

<p align=\"center\">
**Fall 2025 | Math 75650 Statistical Learning II | Date: " current-date "**
</p>

**By:** Emma Naiyue Liang, Ryan Renken, Jaryt Salvo, & Jason Turk

*************

<p align=\"center\" style=\"font-size: 1.5em;\">
**[View our code base on GitHub](https://github.com/adabwana/sp25-m7560-final-project)**
</p>

# Part A: Learning Commons Prediction

## Project Overview

This project develops predictive models for student usage patterns at the BGSU Learning Commons (LC). We focus on two specific prediction challenges: estimating **_visit duration_** and forecasting **_building occupancy_** at the time of check-in. Different modeling approaches were tailored to each task.

## Data Architecture

The analysis uses LC visit data spanning two academic years. Models were trained on data from Fall 2016 - Spring 2017 and evaluated on a holdout set from Fall 2017 - Spring 2018. Features included student demographics, academic metrics (course load, GPA), temporal information (time of day, week of semester), and external data (weather, lunar phase). An observed bias towards senior-class representation in the data was noted.

## Methodological Framework

We employed a structured approach using the **R `tidymodels`** ecosystem. Feature engineering was performed using the **`recipes`** package to create informative predictors from raw data, including transformations for temporal features and standardization of numeric variables.

For both prediction tasks (duration and occupancy), we evaluated several algorithms, including **_Multivariate Adaptive Regression Splines (MARS)_**, **_Random Forest_**, **_XGBoost_**, **_Multi-Layer Perceptron (MLP)_** networks, and **_Gated Recurrent Unit (GRU)_** networks. Hyperparameters for each model were optimized using 5-fold cross-validation specific to the prediction task.

## Implementation Results

Model performance was assessed on the holdout dataset (Fall 2017 - Spring 2018) using Root Mean Squared Error (RMSE) and R-squared (R²).

- **Duration Prediction:** This proved challenging. The best model (XGBoost) achieved an RMSE of 59.9 minutes and an R² of 0.099. Performance was slightly improved by incorporating a weighted average with the training set mean duration.
- **Occupancy Prediction:** This yielded substantially better results. The optimized XGBoost model achieved an RMSE of 1.83 students and an R² of 0.911, demonstrating strong predictive power.

## Technical Infrastructure

The project was implemented in **R**, primarily utilizing the **`tidymodels`** suite (**`recipes`**, **`parsnip`**, **`rsample`**, **`tune`**, **`workflows`**, **`yardstick`**). Model implementations relied on `earth` (MARS), `ranger` (Random Forest), and `xgboost` (XGBoost). Standard packages like **`dplyr`**, **`readr`**, and **`ggplot2`** were used for data handling and visualization. External data integration used `lunar` and `openmeteo`.

## Research Findings

Predicting individual visit duration is inherently difficult, likely due to high variability and skewness in usage patterns, resulting in a low R² (0.10). In contrast, predicting building occupancy was highly successful (R² = 0.91). The features and XGBoost model effectively captured the aggregate patterns influencing concurrent LC usage based on check-in data.

## Future Research Directions

Future work could explore integrating additional environmental factors, applying more complex non-linear or time-series specific models, or developing ensemble methods that combine predictions from multiple models to potentially improve performance, particularly for the duration task.

*Implementation details and comprehensive analysis available in associated documentation.*

-------------------------------------------------------

# Part B: K-means++ Initialization Analysis

## Overview

This section details a comparative analysis between standard K-means and K-means++ initialization methods for clustering.

## Data & Task

The analysis was performed on a dataset consisting of $\\sim 5000$ points sampled from a subset of $\\mathbb{R}^2$. The objective was to partition this data into $k=11$ distinct clusters.

## Methodology Comparison

We compared the performance of standard K-means clustering against K-means utilizing the K-means++ initialization strategy. The K-means++ algorithm aims to select more strategic initial centroids to potentially improve clustering quality and convergence speed compared to random initialization.

## Results

- **Standard K-means:** Converged in 5 iterations with a final Within-Cluster Sum of Squares (WCSS) of 22,824.
- **K-means++ Initialization:** Converged in 8 iterations with a final WCSS of 22,943.

## Findings

For this specific $\\mathbb{R}^2$ dataset and $k=11$, the K-means++ initialization method did not demonstrate a discernible advantage over the standard K-means approach, either visually or based on the WCSS metric. The standard method converged faster and achieved a slightly lower WCSS.

*Visualizations and further details available in associated documentation.*")))
