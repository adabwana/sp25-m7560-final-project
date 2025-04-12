# Load necessary libraries
library(parsnip)
library(dials)
library(ranger)
library(earth)
library(xgboost)
library(dplyr) # Needed for %>% pipe used in spec definitions

# Define model specifications and parameter sets/grids

# --- MARS ---
mars_spec <- mars(mode = "regression", num_terms = tune(), prod_degree = tune()) %>%
  set_engine("earth")

mars_params <- parameters( # Define parameter space
  num_terms(range = c(10L, 50L)),
  prod_degree(range = c(1L, 1L)) # Use range with identical bounds for fixed value
)

# Generate grid using dials::grid_regular
mars_grid <- grid_regular( # Levels define how many points for each param
  mars_params,
  levels = c(num_terms = 5, prod_degree = 1) # Keep prod_degree, level 1 uses the single value
)


# --- Random Forest ---
rf_spec <- rand_forest(mode = "regression", trees = tune(), min_n = tune()) %>%
  set_engine("ranger")

rf_params <- parameters(
  trees(range = c(100L, 200L)),
  min_n(range = c(2L, 5L))
)

# Generate grid using dials::grid_regular
rf_grid <- grid_regular(
  rf_params,
  levels = c(trees = 2, min_n = 2) # Creates 2*2=4 combinations
)

# --- XGBoost ---
xgb_spec <- boost_tree(mode = "regression", trees = tune(), tree_depth = tune(), learn_rate = tune()) %>%
  set_engine("xgboost")

xgb_params <- parameters(
  trees(range = c(100L, 200L)),
  tree_depth(range = c(3L, 9L)),
  learn_rate(range = log10(c(0.01, 0.1))) # dials uses log10 scale for learn_rate
)

# Generate grid using dials::grid_regular
xgb_grid <- grid_regular(
  xgb_params,
  levels = c(trees = 2, tree_depth = 3, learn_rate = 2) # Creates 2*3*2=12 combinations
)

# Store definitions
# Note: Grids are now tibbles (tbl_df) from grid_regular, not data.frames
model_list_duration <- list(
  MARS = list(spec = mars_spec, grid = mars_grid),
  RandomForest = list(spec = rf_spec, grid = rf_grid),
  XGBoost = list(spec = xgb_spec, grid = xgb_grid)
)

# --- Occupancy Model Grids ---
# Since the parameter values are identical for this case, we can reuse the grids
mars_grid_occ <- mars_grid
rf_grid_occ <- rf_grid
xgb_grid_occ <- xgb_grid

# Store occupancy definitions
model_list_occupancy <- list(
  MARS = list(spec = mars_spec, grid = mars_grid_occ),
  RandomForest = list(spec = rf_spec, grid = rf_grid_occ),
  XGBoost = list(spec = xgb_spec, grid = xgb_grid_occ)
)


# Clean up intermediate parameter objects (optional, keep specs/grids/lists)
# rm(mars_params, rf_params, xgb_params)
