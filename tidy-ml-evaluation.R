################################################################################
# Script name: tidy-ml-evaluation
# Author: Arcenis Rojas
# E-mail: arcenis.rojas@tutanota.com
# Date created: 8/28/2021
#
# Script description: Evaluate an ML algorithm using Tidyverse and Tidymodels
#   functions primarily to train and test a model and then to present
#   evaluation metrics and visualizations.
#
################################################################################

# Notes ========================================================================



# TODO Items ===================================================================



# Set up the environment =======================================================

# Clear the workspace
rm(list = ls())

# Store a vector of packages names to install and load
pkg_list <- c(
  "tidyverse", "tidymodels", "magrittr", "janitor", "car", "vip", "fastshap", 
  "pdp", "corrr", "xgboost", "doParallel", "future", "furrr"
)

# Install any packages that are not yet installed
lapply(
  pkg_list, 
  function(p) {
    if (!p %in% installed.packages()[, "Package"]) {
      install.packages(p, dependencies = TRUE)
    }
  }
)

# Also install the 'recipeselectors' package from GitHub
if (!"recipeselectors" %in% installed.packages()[, "Package"]) {
  if (!"devtools" %in% installed.packages()[, "Package"]) {
    install.packages(devtools, dependencies = TRUE)
  }
  
  devtools::install_github("stevenpawley/recipeselectors")
}

# Load packages that will be used throughout the code
library(tidyverse)
library(tidymodels)


# Declare/load helper functions ================================================
source("./helper-functions/tidy-ml-evaluation-funs.R")


################################ Program start #################################

# Load and explore the data ====================================================

# Load the Ames dataset
data(ames)

# Convert month variable to a factor and set the largest category of each
# factor to that factor's base level (for contrast matrix for LM)
housing <- ames |>
  janitor::clean_names() |>
  mutate(
    mo_sold = as_factor(mo_sold),
    across(where(is.factor), ~ fct_drop(.x) |> fct_infreq())
  )

# Ensure that there are no missing variables
any(is.na(housing))

# Plot the number of categories in each categorical variable
housing_categorical <- select(housing, -sale_price) |>
  select_if(is.factor) |>
  summarise(across(everything(), ~ n_distinct(.x, na.rm = TRUE))) |>
  pivot_longer(
    everything(), 
    names_to = "variable", 
    values_to = "num_categories"
  ) |>
  ggplot(aes(x = num_categories, y = fct_reorder(variable, num_categories))) +
  geom_bar(stat = "identity", width = 0.8) +
  labs(
    y = "Variable",
    x = "Number of Categories",
    title = "Number of Categories in Each Factor Variable"
  ) +
  ml_eval_theme()

# Check for near-zero variance
uniqueCut <- select(housing, -sale_price) |>
  select_if(is.factor) |>
  pivot_longer(everything(), names_to = "variable", values_to = "category") |> 
  group_by(variable) |>
  summarise(uniqueCut = (n_distinct(category) * 100) / n(), .groups = "drop")

freqCut <- select(housing, -sale_price) |>
  select_if(is.factor) |>
  pivot_longer(everything(), names_to = "variable", values_to = "category") |>
  count(variable, category, name = "count") |>
  group_by(variable) |>
  slice_max(count, n = 2, with_ties = FALSE) |>
  mutate(rank = c("first", "second")) |>
  ungroup() |>
  select(-category) |>
  pivot_wider(names_from = rank, values_from = count) |>
  mutate(freqCut = first / second)

housing_nzv <- left_join(freqCut, uniqueCut, by = "variable") |>
  mutate(nzv = as.numeric(uniqueCut < 10 & freqCut > 19))

# Check the distributions of categorical variables
housing_numeric <- select(housing, where(is.numeric)) |>
  pivot_longer(everything(), names_to = "variable", values_to = "value") |>
  ggplot(aes(x = variable, y = value)) +
  geom_violin(fill = "gray") +
  facet_wrap(~ variable, scales = "free") +
  labs(
    y = NULL,
    x = NULL,
    title = "Distributions of Numeric Variables"
  ) +
  ml_eval_theme() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

# Visualize relationships among categorical variables
factor_names <- select(housing, -sale_price) |>
  select_if(is.factor) |>
  names()

chi_sq_dat <- crossing(var1 = factor_names, var2 = factor_names) |>
  mutate(
    chi_sq_results = map2(
      var1,
      var2,
      ~ select(housing, any_of(c(.x, .y))) |>
        table() |>
        chisq.test() |>
        broom::tidy()
    )
  ) |>
  unnest(chi_sq_results) |>
  select(var1, var2, p.value) |>
  pivot_wider(names_from = var2, values_from = p.value) |>
  column_to_rownames("var1")

chi_sq_dat[!upper.tri(chi_sq_dat)] <- NA

chi_sq_viz <- chi_sq_dat |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "p.value") |>
  drop_na(p.value) |>
  ggplot(aes(fct_rev(var2), var1, color = p.value)) +
  geom_point(size = 3) +
  scale_color_viridis_c(direction = -1) +
  labs(title = "Chi-square Plot of Categorical Variables", color = "P-value") +
  ml_eval_theme() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.border = element_blank(),
    axis.line = element_line()
  )

# Visualize relationships among numeric variables
corr_viz <- select(housing, -sale_price) |>
  select_if(is.numeric) |>
  corrr::correlate(method = "spearman", use = "pairwise.complete.obs") |>
  corrr::rearrange(absolute = FALSE) |>
  corrr::shave() |>
  corrr::rplot(colors = c("red", "white", "blue")) +
  labs(title = "Correlation Plot of Numeric Variables", color = "Correlation") +
  ml_eval_theme() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.border = element_blank(),
    axis.line = element_line()
  )

# There are many strong relationships among the categorical variables, but only
# a few among the numerical variables


# Create train/test and CV splits ==============================================

# Set the random number seed
set.seed(485)

# Split the data
housing_split <- initial_split(housing, prop = 0.75)
housing_train <- training(housing_split)
housing_test <- testing(housing_split)

# Create the CV dataframe
housing_folds <- vfold_cv(housing_train, v = 10)


# Write pre-processing recipes =================================================

# Write one recipe for XGB and one for LM
xgb_rec <- recipe(housing_train) |>
  update_role(sale_price, new_role = "outcome") |>
  update_role(-has_role("outcome"), new_role = "predictor") |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_YeoJohnson(all_outcomes()) |>
  recipeselectors::step_select_mrmr(
    all_predictors(),
    outcome = "sale_price",
    threshold = 0.9,
    skip = TRUE
  ) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

num_steps <- length(xgb_rec$steps)

lm_rec <- xgb_rec
lm_rec$steps[[num_steps]] <- update(lm_rec$steps[[num_steps]], one_hot = FALSE)


# Train models =================================================================

# Get the number of features in the training data for XGB tuning grid
n_features <- bake(prep(xgb_rec), new_data = housing_train) |>
  ncol() |>
  magrittr::subtract(1) |>
  sqrt()

# Create a tuning grid for XGB
set.seed(55)
xgb_grid <- grid_random(
  mtry(c(1, floor(n_features))),   # Range of number of features to try
  trees(),   # Range of number of trees
  learn_rate(range = c(-7, 1)),   # Learning rate
  loss_reduction(),
  size = 50
)

# Create model objects for both XGB and LM
xgb_mod <- boost_tree(
  mtry = tune(), 
  trees = tune(), 
  learn_rate = tune(),
  loss_reduction = tune(),
  mode = "regression"
) |>
  set_engine("xgboost")

lm_mod <- linear_reg(mode = "regression") |> set_engine("lm")

# Create workflow objects for tuning
xgb_wflow <- workflow() |> add_recipe(xgb_rec) |> add_model(xgb_mod)
lm_wflow <- workflow() |> add_recipe(lm_rec) |> add_model(lm_mod)

# Register and set up the parallel backend
cl <- parallel::makePSOCKcluster(parallel::detectCores(logical = FALSE) - 1)
doParallel::registerDoParallel(cl)
parallel::clusterEvalQ(cl, library(recipeselectors))
parallel::clusterEvalQ(cl, set.seed(853))

# Tune both models
xgb_tune <- tune_grid(
  xgb_wflow,
  grid = xgb_grid,
  resamples = housing_folds,
  metrics = metric_set(rmse, rsq_trad)
)

lm_tune <- fit_resamples(
  lm_wflow,
  resamples = housing_folds,
  metrics = metric_set(rmse, rsq_trad)
)

# Leaving the parallel back end open because it's necessary for last_fit()


# Extract and fit the best models ==============================================

# Get the best models from each tuning process
best_models <- list(xgb = xgb_tune, lm = lm_tune) |>
  map(select_best, metric = "rmse")

# Collect tuning metrics for each tuning process
tune_metrics <- list(xgb = xgb_tune, lm = lm_tune) |>
  map(collect_metrics)

# Fit the models with the best parameters to the entire training dataset
final_wflows <- map2(
  list(xgb = xgb_wflow, lm = lm_wflow),
  best_models,
  ~ finalize_workflow(.x, .y) |> fit(data = housing_train)
)

# Evaluate each model with the test data and store 'last_fit' objects
set.seed(485)
final_wflow_evals <- map(
  final_wflows, 
  last_fit, 
  split = housing_split,
  metrics = metric_set(rmse, mae, mape, rsq_trad)
)

# Temporarily return to sequential processing
doParallel::stopImplicitCluster()

# Generate a dataframe of model metrics to compare the two models
model_metrics <- final_wflow_evals |>
  map2_df(
    names(final_wflow_evals),
    ~ collect_metrics(.x) |> 
      select(.metric, .estimate) |> 
      pivot_wider(names_from = .metric, values_from = .estimate) |>
      mutate(algorithm = .y) |>
      relocate(algorithm)
  )


# Generate Variable Importance and SHAP values =================================

# Get matrices of training features for each model
training_features <- final_wflows %>% 
  map(~ pull_workflow_mold(.x) %>% pluck("predictors"))

# Initiate a multicore plan 
future::plan("cluster", workers = cl)

# Get SHAP values
shap <- final_wflows %>% 
  furrr::future_map(
    get_shap, 
    .progress = TRUE, 
    .options = furrr::furrr_options(seed = 44)
  )

# Get SHAP variable importance features
var_imp <- shap %>%
  map(get_shap_imp)

# Go back to sequential processing
future::plan(future::sequential)

# Close the connection to the cluster
parallel::stopCluster(cl)

# Create SHAP summaries containing data frames and plots
shap_summary_plots <- list(
  var_imp, 
  shap, 
  training_features
) |>
  pmap(get_shap_summary, max_features = 10) %>%
  map2(c("XG Boost", "Linear Regression"), plot_shap_summary)

gr_liv_area_pdp <- final_wflows |>
  map2(c("XG Boost", "Linear Regression"), plot_pdp, pred_var = gr_liv_area)

obs_2409_contrib <- shap |>
  map2(
    c("XG Boost", "Linear Regression"), 
    get_contributions, 
    rnum = 2049, 
    nfeat = 15 
  )

obs_2409_contrib$xgb$contrib_plot
obs_2409_contrib$lm$contrib_plot
