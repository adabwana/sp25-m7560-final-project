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
  num_terms(range = c(7L, 15L)), # centered around 10
  prod_degree(range = c(1L, 1L)) # Keep fixed
)

# Generate grid using dials::grid_regular
mars_grid <- grid_regular( # Levels define how many points for each param
  mars_params,
  levels = c(num_terms = 9, prod_degree = 1) # 9 levels in the range
)


# --- Random Forest ---
rf_spec <- rand_forest(
  mode = "regression", trees = tune(), min_n = tune(), mtry = tune()
) %>%
  set_engine("ranger")

# Update RF parameter ranges based on duration tuning results (high RMSE)
rf_params <- parameters(
  trees(range = c(300L, 325L)), # Centered around 200
  min_n(range = c(15L, 25L)), # Focused around 20
  mtry(range = c(20L, 25L)) # Exploring 20 range
)

# Generate grid using dials::grid_regular
rf_grid <- grid_regular(
  rf_params,
  levels = c(trees = 2, min_n = 3, mtry = 2) # 2*3*2 = 12 combinations
)

# --- XGBoost ---
xgb_spec <- boost_tree(
  mode = "regression", trees = tune(), tree_depth = tune(), learn_rate = tune(),
  min_n = tune(), mtry = tune()
) %>%
  set_engine("xgboost")

# Update XGBoost parameter ranges based on duration tuning results (high RMSE)
xgb_params <- parameters(
  trees(range = c(75L, 100L)), # Exploring 100-300 range
  tree_depth(range = c(15L, 21L)), # Exploring 20 range
  learn_rate(range = log10(c(0.05, 0.05))), # Narrow focus around 0.05
  min_n(range = c(10L, 15L)), # Focus tightly around 10
  mtry(range = c(12L, 15L)) # Exploring 5-20 range
)

# Generate grid using dials::grid_regular
xgb_grid <- grid_regular(
  xgb_params,
  levels = c(trees = 3, tree_depth = 3, learn_rate = 1, min_n = 2, mtry = 2) # 2*3*1*2*3 = 36 combinations. DONT GO MORE THAN 36!!
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
    num_terms(range = c(120L, 130L)), # best 122
    prod_degree(range = c(1L, 1L))
  ),
  levels = c(num_terms = 10, prod_degree = 1) # Explore 6 points in the new range
)

# Random Forest Occupancy Grid: Center around best duration results
rf_grid_occ <- grid_regular(
  parameters(
    trees(range = c(250L, 350L)), # Centered around 300
    min_n(range = c(2L, 3L)), # Centered around 2-4
    mtry(range = c(40L, 45L)) # Centered around 35
  ),
  levels = c(trees = 3, min_n = 2, mtry = 2) # 3*2*2 = 12 combinations
)

# XGBoost Occupancy Grid: Focus on promising region
xgb_grid_occ <- grid_regular(
  parameters(
    trees(range = c(350L, 450L)), # Centered around 375
    tree_depth(range = c(6L, 8L)), # Centered around 7
    learn_rate(range = log10(c(0.1, 0.1))), # Explore around 0.1
    min_n(range = c(2L, 3L)), # Around 2-3
    mtry(range = c(30L, 35L)) # Centered around 30
  ),
  levels = c(trees = 3, tree_depth = 3, learn_rate = 1, min_n = 2, mtry = 2) # 3*3*1*2*2 = 36 combinations. DONT GO MORE THAN 36!!
)

# Store occupancy definitions
model_list_occupancy <- list(
  MARS = list(spec = mars_spec, grid = mars_grid_occ),
  RandomForest = list(spec = rf_spec, grid = rf_grid_occ),
  XGBoost = list(spec = xgb_spec, grid = xgb_grid_occ)
)


# Clean up intermediate parameter objects (optional, keep specs/grids/lists)
# rm(mars_params, rf_params, xgb_params)
