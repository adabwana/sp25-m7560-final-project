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
### Predicting Learning Commons Usage: Duration and Occupancy

<p align=\"center\">
**Fall 2024 | Math 7550 Statistical Learning I | Date: " current-date "**
</p>

**By:** Emma Naiyue Liang, Ryan Renken, Jaryt Salvo, & Jason Turk

*************

<p align=\"center\" style=\"font-size: 1.5em;\">
**[View our code base on GitHub](https://github.com/adabwana/f24-m7550-final-project)**
</p>

## Project Overview

In our project, we implement systematic prediction methodologies for analyzing student utilization patterns within the BGSU Learning Commons (LC). Through rigorous statistical learning approaches, we address two distinct prediction challenges: **_visit duration estimation_** and **_occupancy forecasting_**. The implementation leverages specialized modeling architectures to capture the unique characteristics of each prediction task.

## Data Architecture

The research framework employs a comprehensive  dataset spanning two academic years. The training corpus encompasses Fall 2016 through Spring 2017, providing the foundation for model development and parameter optimization. Our validation framework utilizes subsequent academic year data (Fall 2017 - Spring 2018) to assess model generalization and stability. The feature space integrates **_demographic indicators_**, **_academic metrics_**, and **_temporal patterns_**, while accounting for an observed senior-class representation bias in the underlying data collection process.

## Methodological Framework

Our preprocessing architecture implements systematic feature engineering across multiple domains. The temporal component decomposes visit patterns into hierarchical time scales, capturing daily rhythms, weekly cycles, and semester-long trends. Academic context modeling integrates course-level hierarchies with performance metrics, while our statistical standardization protocol employs **_RobustScaler_** methodology for outlier resilience.

The modeling architecture employs task-specific approaches while maintaining shared foundational components. The core implementation utilizes **_PenalizedSplines_** as the primary architecture, supplemented by Ridge and Lasso regression variants for linear modeling and KNN for local pattern capture. Duration-specific modeling incorporates Penalized Log-Normal GLM for pattern modeling, while occupancy prediction employs specialized Poisson and Weibull architectures for count-based forecasting.

## Implementation Results

The **_duration prediction framework_** achieves an RMSE of 59.47 minutes with an R² of 0.059, utilizing PenalizedSplines with optimized parameters (Ridge α: 14.38, spline degree: 3, knot count: 15). 

The **_occupancy prediction system_** demonstrates enhanced predictive capacity with an RMSE of 3.64 students and R² of 0.303, employing similar architectural components with task-specific parameter optimization.

## Technical Infrastructure

Our implementation leverages the Python scientific computing ecosystem, with **_scikit-learn_** providing the core modeling framework and **_MLflow_** enabling systematic experiment tracking. The data processing pipeline integrates pandas and numpy for efficient computation, while visualization capabilities combine matplotlib, seaborn, and plotly for comprehensive analysis. The training infrastructure implements automated parameter optimization through grid search methodology, while maintaining systematic version control through MLflow's tracking capabilities.

## Research Findings

Duration prediction presents significant challenges due to high variance in visit patterns and right-skewed distributions. The limited predictive capacity (R² = 0.059) suggests underlying complexity in individual visit duration behaviors. In contrast, occupancy prediction demonstrates more robust performance (R² = 0.303), successfully capturing usage patterns and providing reliable concurrent usage estimates.

## Future Research Directions

This research framework enables several promising extensions. Environmental factor integration, particularly weather patterns, could enhance predictive capacity. Deep pattern analysis through advanced non-linear methodologies and specialized time series approaches warrant investigation. Additionally, ensemble architectures combining multiple modeling paradigms present opportunities for performance enhancement.

*Implementation details and comprehensive analysis available in associated documentation.*")))
