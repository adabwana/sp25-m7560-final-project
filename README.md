# M7550: Final Project

### Predicting Learning Commons Usage: Duration and Occupancy

**By:** Emma Naiyue Liang, Ryan Renken, Jaryt Salvo, & Jason Turk

**Fall 2024 | Math 7550 Statistical Learning I**

*************

## **[Please click HERE to view our Report Submission](https://adabwana.github.io/f24-m7550-final-project/).**

## Project Overview

In our project, we implement systematic prediction methodologies for analyzing student utilization patterns within the BGSU Learning Commons (LC). Through rigorous statistical learning approaches, we address two distinct prediction challenges: **_visit duration estimation_** and **_occupancy forecasting_**. The implementation leverages specialized modeling architectures to capture the unique characteristics of each prediction task.

## Data Architecture

The research framework employs a comprehensive dataset spanning two academic years. The training corpus encompasses Fall 2016 through Spring 2017, providing the foundation for model development and parameter optimization. Our validation framework utilizes subsequent academic year data (Fall 2017 - Spring 2018) to assess model generalization and stability. The feature space integrates **_demographic indicators_**, **_academic metrics_**, and **_temporal patterns_**, while accounting for an observed senior-class representation bias in the underlying data collection process.

## Methodological Framework

Our preprocessing architecture implements systematic feature engineering across multiple domains:

- Temporal component decomposition into hierarchical time scales
- Academic context modeling with course-level hierarchies
- Statistical standardization using **_RobustScaler_** methodology

The modeling architecture employs:
- **_PenalizedSplines_** as the primary architecture
- Ridge and Lasso regression variants
- KNN for local pattern capture
- Penalized Log-Normal GLM for duration modeling
- Specialized Poisson and Weibull architectures for occupancy forecasting

## Implementation Results

- **Duration Prediction**: RMSE of 59.47 minutes (R² = 0.059)
  - PenalizedSplines with optimized parameters
  - Ridge α: 14.38
  - Spline degree: 3
  - Knot count: 15

- **Occupancy Prediction**: RMSE of 3.64 students (R² = 0.303)

## Technical Infrastructure

The project utilizes:
- Python scientific computing ecosystem
- **_scikit-learn_** for core modeling
- **_MLflow_** for experiment tracking
- pandas and numpy for computation
- matplotlib, seaborn, and plotly for visualization
- Clay for notebook rendering
- R for statistical analysis
- Quarto for HTML generation

### Project Structure

The workbook is organized into several key sections:

1. **Feature Engineering**
   - Temporal Features (time decomposition, custom categories)
   - Academic Context (course levels, GPA, student standing)
   - Course & Major Analysis (keyword categorization, grouping)
   - Usage Patterns (frequency, volume, group dynamics)
   - Data Quality (duration/occupancy validation)

2. **Models & Pipelines**
   - Cross-Validation (KFold, Rolling, Expanding windows)
   - Algorithms (Ridge/Lasso, PenalizedSplines, KNN)
   - Pipeline Variants (Vanilla, Interaction, Dimensionality Reduction)

3. **Evaluation**
   - Duration Model (RMSE: 59.47, R²: 0.059)
   - Occupancy Model (RMSE: 3.64, R²: 0.303)
   - Model Diagnostics (distributions, residuals)
   - Technical Challenges & Comparisons

4. **Predictions**
   - Duration prediction analysis
   - Occupancy forecasting results

5. **Appendix**
   - Technical details
   - Supplementary analyses

*Implementation details and comprehensive analysis available in the interactive workbook.*

## Getting Started

To work on this project, you'll need to install Docker and VSCode (or fork). Using Dev Containers allows you to develop consistently across any operating system (Windows, macOS, or Linux) without worrying about installing dependencies or configuring your local environment. The containerized development environment ensures that all team members work with identical setups, regardless of their host machine.

### Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) - Provides containerization
- [Visual Studio Code](https://code.visualstudio.com/) - Code editor with Dev Containers support

### Step-by-Step Setup

1. Install Docker Desktop and VSCode on your system.

2. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VSCode. This enables VSCode to develop inside containers.

3. Clone and open the repository:
   ```bash
   cd Documents/projects  # or your preferred directory
   git clone https://github.com/adabwana/f24-m7550-final-project.git
   cd f24-m7550-final-project
   code .
   ```

4. When VSCode opens, look for the "Open in Dev Container" popup in the bottom right. Alternatively, press `Ctrl + Shift + P`, type "Dev Containers: Open Folder in Container", and select the project folder.

5. Choose your development environment:
   - R container: For feature engineering and visualization
   - Python container: For modeling and additional plotting
   - LaTeX container: For presentation development
   - Clojure container: For notebook rendering

The Dev Container will automatically:
- Set up the correct version of R, Python, or other required tools as per the selected development environment
- Install all necessary packages and dependencies
- Configure the development environment consistently
- Isolate the project environment from your local system

This approach ensures reproducibility and eliminates "it works on my machine" issues when collaborating.

### Note

Due to the project's limited duration, full automation of the workflow is not implemented. To view and run the models, please follow these sequential steps:

1. **R Environment (Feature Engineering)**:
   - First, open the project in the R development container (described above)
   - Run `src/r/feature_engineering.r` to generate the engineered feature CSVs
   - This step is required before any model training

2. **Switch to Python Environment**:
   - After feature engineering, you'll need to switch environments
   - Press `Ctrl + Shift + P` and start typing "Reopen Locally" and select itß
   - Press `Ctrl + Shift + P` again and start typing "Dev Containers: Reopen in Container" and select it
   - Select the Python development environment

3. **Model Training**:
   - With the generated CSVs from step 1 (in `data/processed`)
   - Navigate to `src/python/test_train`
   - Run the training scripts to view model results logged in MLflow
   - If MLflow did not open in your browser, you can open it on [`localhost:5000`](http://localhost:5000). Check it out, MLflow is a powerful tool

Note that this workflow requires manual environment switching due to the separate R and Python dependencies. But hopefully the process of bouncing back and forth between development environment and local will allow for deeper understanding of Docker, Dev Containers, and the project.

Subsequent testing, visualizations, predictions, etc. are in the project and are left for the curious reader to explore, if they dare.