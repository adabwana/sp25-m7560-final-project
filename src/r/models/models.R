# Load necessary libraries
library(parsnip)
library(dials)
library(ranger)
library(earth)
library(xgboost)
library(dplyr) # Needed for %>% pipe used in spec definitions

# Define model specifications and parameter sets/grids

# --- MARS ---
mars_spec <- mars(
  mode = "regression", num_terms = tune(), prod_degree = tune()
) %>%
  set_engine("earth")

# Update MARS parameter range based on duration tuning results (high RMSE)
mars_params <- parameters( # Define parameter space
  num_terms(range = c(1L, 20L)), # Based on flat performance in 20-50 range
  prod_degree(range = c(1L, 1L)) # Keep fixed
)

# Generate grid using dials::grid_regular
mars_grid <- grid_regular( # Levels define how many points for each param
  mars_params,
  levels = c(num_terms = 10, prod_degree = 1) # 5 levels in the range
)


# --- Random Forest ---
rf_spec <- rand_forest(
  mode = "regression", trees = tune(), min_n = tune(), mtry = tune()
) %>%
  set_engine("ranger")

# Update RF parameter ranges based on duration tuning results (high RMSE)
rf_params <- parameters(
  trees(range = c(150L, 250L)), # Centered around 200
  min_n(range = c(6L, 15L)), # Focused around 6-10
  mtry(range = c(10L, 25L)) # Exploring 5-20 range
)

# Generate grid using dials::grid_regular
rf_grid <- grid_regular(
  rf_params,
  levels = c(trees = 3, min_n = 4, mtry = 4) # 3*4*4 = 48 combinations
)

# --- XGBoost ---
xgb_spec <- boost_tree(
  mode = "regression", trees = tune(), tree_depth = tune(), learn_rate = tune(),
  min_n = tune(), mtry = tune()
) %>%
  set_engine("xgboost")

# Update XGBoost parameter ranges based on duration tuning results (high RMSE)
xgb_params <- parameters(
  trees(range = c(100L, 300L)), # Exploring 100-300 range
  tree_depth(range = c(3L, 7L)), # Exploring 3-6 range
  learn_rate(range = log10(c(0.08, 0.12))), # Narrow focus around 0.1
  min_n(range = c(8L, 12L)), # Focus tightly around 10
  mtry(range = c(5L, 25L)) # Exploring 5-20 range
)

# Generate grid using dials::grid_regular
xgb_grid <- grid_regular(
  xgb_params,
  levels = c(trees = 3, tree_depth = 3, learn_rate = 2, min_n = 3, mtry = 4) # 3*3*2*3*4 = 216 combinations
)

# Store definitions
# Note: Grids are now tibbles (tbl_df) from grid_regular, not data.frames
model_list_duration <- list(
  MARS = list(spec = mars_spec, grid = mars_grid),
  RandomForest = list(spec = rf_spec, grid = rf_grid),
  XGBoost = list(spec = xgb_spec, grid = xgb_grid)
)

# --- Occupancy Model Grids ---
# Adjusting grids based on duration tuning results

# MARS Occupancy Grid: Explore higher num_terms
mars_grid_occ <- grid_regular(
  parameters(
    num_terms(range = c(40L, 90L)), # Adjusted range based on results
    prod_degree(range = c(1L, 1L))
  ),
  levels = c(num_terms = 6, prod_degree = 1) # Explore 6 points in the new range
)

# Random Forest Occupancy Grid: Center around best duration results
rf_grid_occ <- grid_regular(
  parameters(
    trees(range = c(150L, 250L)), # Centered around 200
    min_n(range = c(2L, 8L)), # Centered around 2-6
    mtry(range = c(15L, 25L)) # Centered around 20
  ),
  levels = c(trees = 3, min_n = 3, mtry = 3) # 3*3*3 = 27 combinations
)

# XGBoost Occupancy Grid: Focus on promising region
xgb_grid_occ <- grid_regular(
  parameters(
    trees(range = c(200L, 400L)), # Centered around 300
    tree_depth(range = c(5L, 7L)), # Centered around 6
    learn_rate(range = log10(c(0.05, 0.15))), # Explore around 0.1
    min_n(range = c(2L, 10L)), # Around 2-6
    mtry(range = c(15L, 25L)) # Centered around 20
  ),
  levels = c(trees = 3, tree_depth = 3, learn_rate = 2, min_n = 3, mtry = 3) # 3*3*2*3*3 = 162 combinations
)

# Store occupancy definitions
model_list_occupancy <- list(
  MARS = list(spec = mars_spec, grid = mars_grid_occ),
  RandomForest = list(spec = rf_spec, grid = rf_grid_occ),
  XGBoost = list(spec = xgb_spec, grid = xgb_grid_occ)
)


# Clean up intermediate parameter objects (optional, keep specs/grids/lists)
# rm(mars_params, rf_params, xgb_params)
