# M7560: Final Project

### Predicting Learning Commons Usage: Duration and Occupancy

**By:** Naiyue Liang (Emma), Ryan Renken, Jaryt Salvo, & Jason Turk

**Spring 2025 | Math 7560 Statistical Learning II**

*************

## **[Please click HERE to view our Report Submission](https://adabwana.github.io/sp25-m7560-final-project/).**

## Project Overview

In this project, we look at how students use the BGSU Learning Commons (LC). Using statistical learning methods, we tackle two main prediction tasks: figuring out how long students stay (**_visit duration estimation_**) and how many students are there at a given time (**_occupancy forecasting_**). We use different modeling approaches designed to handle the specific details of each task.

## Data Architecture

We used a large dataset covering two full school years. The data from Fall 2016 through Spring 2017 was used for training our models. We then tested how well the models worked on new data from the following school year (Fall 2017 - Spring 2018). The information we used includes student **_demographics_**, **_academic details_**, **_when students visited_**, plus **_external data_** like weather and moon phases. We also kept in mind that the way the data was originally collected might have included more seniors than other classes.

## Methodological Framework

Our preprocessing architecture, implemented using the R `recipes` package, includes:
- Creating new features across different areas (Temporal, Academic, Visit, Course, Student).
- Transformation of raw features into informative derived features.
- Handling of time variables, imputation of missing values, management of novel factor levels, dummy variable creation, and normalization.

As part of exploratory analysis, K-means++ initialization was compared to standard K-means for a sample dataset, showing no clear advantage in that specific test.

The modeling architectures explored include:
- Multivariate Adaptive Regression Splines (MARS)
- Random Forest
- XGBoost
- Gated Recurrent Unit (GRU) networks
- Multi-Layer Perceptron (MLP) networks

## Implementation Results

- **Best Model**: XGBoost performed best on the holdout set for both tasks.
- **Duration Prediction**: Holdout RMSE of 59.9 minutes (R² = 0.099)
  - Performance further improved post-hoc using a weighted average (75% prediction, 25% training mean).
  - Best XGBoost Parameters: Trees=75, Depth=21, LR=0.05, MinNode=15, Mtry=15.
- **Occupancy Prediction**: Holdout RMSE of 1.83 students (R² = 0.911)
  - Best XGBoost Parameters: Trees=450, Depth=8, LR=0.1, MinNode=2, Mtry=35.

## Technical Infrastructure

The project utilizes:
- R for feature engineering (`recipes`, `lunar`, `openmeteo`) and statistical analysis.
- Python scientific computing ecosystem for modeling (`scikit-learn`, `xgboost`, potentially `tensorflow`/`pytorch`).
- **_MLflow_** for experiment tracking.
- pandas and numpy for computation.
- matplotlib, seaborn, and plotly for visualization.
- LaTeX for presentation development (`beamer`).
- Docker for containerized development environments.

### Project Structure

The workbook is organized into several key sections:

1. **Feature Engineering**
   - Temporal Features (time of day, day of week, week, semester)
   - Academic Context (course levels, GPA categories, credit load)
   - Course & Major Analysis (subject areas, level progression, course mix)
   - Student Characteristics (major groups, class standing, academic progress)
   - Visit Patterns (duration patterns, group sizes, frequency)
   - External Data (lunar phases, weather metrics)

2. **Models & Pipelines**
   - R `recipes` preprocessing pipeline.
   - Cross-Validation (5-fold used for final model evaluation).
   - Algorithms Explored (MARS, Random Forest, XGBoost, GRU, MLP).

3. **Evaluation**
   - Holdout Set Performance (RMSE, R²)
   - Best Model: XGBoost
   - Duration Model (RMSE: 59.9, R²: 0.099)
   - Occupancy Model (RMSE: 1.83, R²: 0.911)
   - Weighted average technique for duration.
   - Model Diagnostics (distributions, residuals).

4. **Predictions**
   - Analysis of predictions from the best models.

5. **Appendix**
   - K-means++ comparison.
   - Technical details.
   - Supplementary analyses.

*Implementation details and comprehensive analysis available in the interactive workbook and presentation.*

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
   - R container: For feature engineering and visualization (`recipes`)
   - Python container: For modeling (`xgboost`, deep learning) and additional plotting
   - LaTeX container: For presentation development (`beamer`)
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
   - Run `src/r/feature_engineering.r` to generate the engineered feature CSVs using the `recipes` package.
   - This step is required before any model training.

2. **Switch to Python Environment**:
   - After feature engineering, you'll need to switch environments.
   - Press `Ctrl + Shift + P` and start typing "Reopen Locally" and select it.
   - Press `Ctrl + Shift + P` again and start typing "Dev Containers: Reopen in Container" and select it.
   - Select the Python development environment.

3. **Model Training**:
   - With the generated CSVs from step 1 (in `data/processed`).
   - Navigate to `src/python/test_train` (or relevant modeling scripts).
   - Run the training scripts (e.g., for XGBoost, GRU, MLP) to view model results logged in MLflow.
   - If MLflow did not open in your browser, you can open it on [`localhost:5000`](http://localhost:5000). Check it out, MLflow is a powerful tool.

Note that this workflow requires manual environment switching due to the separate R and Python dependencies. But hopefully the process of bouncing back and forth between development environment and local will allow for deeper understanding of Docker, Dev Containers, and the project.

Subsequent testing, visualizations, predictions, etc. are in the project and are left for the curious reader to explore, if they dare.