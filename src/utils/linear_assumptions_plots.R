library(broom)
library(ggplot2)
library(dplyr)

# Set ggplot2 theme
theme_set(theme_minimal())

##### Linear Model Assumptions #####
### 1. Check for linearity between different variables ###
check_linearity_independence <- function(model) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model)
  
  # Create the plot
  ggplot(df, aes(x = .fitted, y = .resid)) +
    geom_point() +
    geom_smooth(span = 1, se = FALSE, color = "gray66", lty = 2) +
    geom_hline(yintercept = 0, col = "red", lty = 2) +
    labs(title = "Check for Independence of \nRandom Error and Linearity",
         x = "Fitted Values",
         y = "Residuals")
}

##### 2. Check for normality of random error #####
check_normality_qq <- function(model) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model)
  
  # Create the Q-Q plot
  ggplot(df, aes(sample = .resid)) +
    stat_qq() +
    stat_qq_line(color = 'red', linetype = 2) +
    ggtitle("Normal Q-Q Plot of Residuals") +
    xlab("Theoretical Quantiles") +
    ylab("Sample Quantiles")
}

### 3. Check for zero mean and constant variance of random error ###
check_homoscedasticity <- function(model) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model)
  
  # Calculate standardized residuals
  std_resid <- rstandard(model)
  
  # Create the Scale-Location plot
  ggplot(df, aes(x = .fitted, y = sqrt(abs(std_resid)))) +
    geom_point() +
    geom_smooth(span = 1, se = FALSE, color = "gray66", lty = 2) +
    geom_hline(yintercept = mean(sqrt(abs(std_resid))), col = "red", lty = 2) +
    labs(title = "Scale-Location",
         x = "Fitted Values",
         y = "sqrt(abs(Standardized Residuals))")
}

### 4. Check for independence of random error ###
check_independence <- function(model, sort_var) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model)
  
  # Get regressor variables
  regress.vars <- names(coef(model))[-1]
  
  # Create df2
  df2 <- df[c(regress.vars, ".resid")] %>% 
    tidyr::pivot_longer(cols = regress.vars, names_to = "statistic", values_to = "xvalue")
  
  # Filter for the specified sort variable
  df2_filtered <- df2 %>% 
    dplyr::filter(statistic == sort_var) %>%
    dplyr::arrange(-xvalue)
  
  # Create the plot
  ggplot(df2_filtered,
         aes(x = 1:nrow(df2_filtered), y = .resid)) +
    geom_point() +
    geom_smooth(span = 1, se = FALSE, color = "gray66", lty = 2) +
    geom_hline(yintercept = 0, col = "red", lty = 2) +
    labs(title = paste("Check for Independence \n Residuals sorted by", sort_var),
         x = "Row Numbers",
         y = "Residuals")
}

# # scatter plot of residuals sorted by hp
# ggplot(df2 %>% arrange(-xvalue),
#        aes(x = 1:nrow(df2), y = .resid)) +
#   geom_point() +
#   geom_smooth(span = 1, se = F, color = "gray66", lty = 2) +
#   geom_hline(yintercept = 0, col = "red", lty = 2) +
#   labs(title = "Check for Independence \n Residuals sorted by hp",
#        x = "Row Numbers",
#        y = "Residuals")

### Final diagnostic plot often presented, too
check_observed_vs_predicted <- function(model, response) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model, data = model$model)
  
  # Create the plot
  ggplot(df, aes(x = .fitted, y = .data[[response]])) +
    geom_point() +
    geom_smooth(span = 1, se = FALSE, color = "gray66", lty = 2) +
    geom_abline(intercept = 0, slope = 1, col = "red", lty = 2) +
    labs(title = "Observed vs Predicted Values",
         x = "Fitted Values",
         y = "Actual Values")
}

# ggplot(df, aes(x = .fitted, y = get(response))) +
#   geom_point() +
#   geom_smooth(span = 1, se = F, color = "gray66", lty = 2) +
#   geom_abline(intercept  = 0, slope = 1, col = "red", lty = 2) +
#   labs(title = "Observed vs Predicted Values ",
#        x = "Fitted Values",
#        y = "Actual Values")

check_residuals_vs_leverage <- function(model) {
  # Pre-process: build a dataframe convenient for ggplot
  df <- broom::augment(model, data = model$model)
  
  # Calculate leverage values
  leverage <- hatvalues(model)
  
  # Create the plot
  ggplot(df, aes(x = leverage, y = .std.resid)) +
    geom_point() +
    geom_smooth(span = 1, se = FALSE, color = "gray66", lty = 2) +
    geom_hline(yintercept = 0, col = "red", lty = 2) +
    labs(title = "Residuals vs Leverage",
         x = "Leverage",
         y = "Standardized Residuals")
}

