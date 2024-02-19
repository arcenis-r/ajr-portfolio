# All the code necessary to write Forecasting with {modeltime} - Part IV is
# here.

# TODO: separate the code for building the workflowsets for part 5
# TODO: consider adding a neural-net workflow and an STLM workflow; run the
#       neural-net on it's own first to test the computation time

# Load required packages
library(tidyverse)
library(timetk)
library(tidymodels)
library(modeltime)
library(multidplyr)

# Resolve common namespace conflicts about function names
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")
conflicted::conflict_prefer("between", "dplyr")
if ("tidymodels" %in% .packages()) tidymodels_prefer()

econ_data <- read_rds("./common-resources/econ-data.Rds")

econ_splits_global <- econ_data |>
  time_series_split(
    assess = "2 year",
    cumulative = TRUE,
    date_var = date   
  )


# Recipes ================

arima_rec1 <- recipe(hpi ~ date, data = training(econ_splits_global))

arima_rec2 <- recipe(
  hpi ~ date + unemp_rate,
  data = training(econ_splits_global)
) |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>
  step_lag(unemp_rate, lag = c(1, 3, 6))

rec3_dep_vars <- econ_data |>
  select(date, unemp_rate:population) |>
  select(!c(educ_hs_less, status_married, age_36_65)) |>
  names() |>
  str_flatten(collapse = " + ")

arima_rec3 <- recipe(
  formula(str_c("hpi", rec3_dep_vars, sep = " ~ ")),
  data = training(econ_splits_global)
) |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>
  step_lag(unemp_rate, population, lag = c(1, 3, 6))

pb_rec <- recipe(
  hpi ~ .,
  data = training(econ_splits_global)
) |>
  update_role(city, new_role = "ID") |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>
  step_timeseries_signature(date) |>
  step_nzv(all_predictors()) |>
  step_lag(unemp_rate, population, lag = c(1, 3, 6))

steps2rmv <- map(pb_rec$steps, \(x) pluck(x, "id")) |>
  unlist() |>
  str_detect("lag") |>
  which()

nnet_rec <- pb_rec

nnet_rec$steps <- nnet_rec$steps[-steps2rmv]


# Model specs =========================

# ARIMA - default
arima_default <- arima_reg() |> set_engine("auto_arima")

# ARIMA - tune for the length of the season
arima_spec_grid <- tibble(seasonal_period = seq(9, 18, by = 3)) |>
  create_model_grid(
    f_model_spec = arima_reg,
    engine_name = "auto_arima",
    mode = "regression"
  )

# Exponential smoothing - default
ets_default <- exp_smoothing() |> set_engine("ets")

# Exponential smoothing - using Holt Winters method
# First create vectors to use for the features to tune
set.seed(40)
alpha_vals <- runif(5, 0.01, 0.5) *
  sample(c(1, 0.1, 0.01, .001), 5, replace = TRUE)
beta_vals <- runif(5, 0.01, 0.2) * sample(c(1, 0.1, 0.01), 5, replace = TRUE)
gamma_vals <- map_dbl(
  alpha_vals,
  \(x) runif(1, 0, 1 - x)
)

# Create the grid
ets_spec_grid <- expand_grid(
  smooth_level = alpha_vals[-1],
  smooth_trend = beta_vals,
  smooth_seasonal = gamma_vals,
  seasonal_period = seq(9, 18, by = 3)
) |>
  create_model_grid(
    f_model_spec = exp_smoothing,
    engine_name = "ets",
    mode = "regression",
    error = "auto",
    trend = "auto",
    season = "auto"
  )

tbats_spec_grid <- tibble(seasonal_period_2 = c(NULL, seq(6, 18, by = 3))) |>
  create_model_grid(
    f_model_spec = seasonal_reg,
    engine_name = "tbats",
    mode = "regression",
    seasonal_period_1 = 12
  )

stlm_spec_grid <- expand_grid(
  seasonal_period_2 = c(NULL, seq(6, 18, by = 3))
  # engine_name = c("stlm_arima", "stlm_ets")
) |>
  create_model_grid(
    f_model_spec = seasonal_reg,
    engine_name = "stlm_arima",
    mode = "regression",
    seasonal_period_1 = 12
  )

# Prophet Boost - default
pb_default <- prophet_boost() |> set_engine("prophet_xgboost")

# Prophet Boost - grid for tuning
# Get the number of features for the 'mtry' hyperparameter
num_features <- pb_rec |> prep() |> juice() |> ncol()

# Create a grid for the ProphetBoost algorithm
pb_spec_grid <- expand_grid(
  learn_rate = c(.1, .05, .025, .01, .001),
  changepoint_num = c(3, 5, 10, 10),
  trees = c(100, 500, 1000, 1500)
) |>
  create_model_grid(
    f_model_spec = prophet_boost,
    engine_name = "prophet_xgboost",
    mode = "regression",
    growth = "linear",
    seasonality_yearly = TRUE,
    mtry = num_features / 3
  )

nnet_default <- nnetar_reg() |> set_engine("nnetar")

nnet_spec_grid <- expand_grid(
  non_seasonal_ar = c(NULL, 1:3),
  seasonal_ar = c(NULL, 1),
  hidden_units = 1:5,
  num_networks = c(5, 10, 20),
  epochs = c(20, 50, 100),
  penalty = c(0.01, 0.05, 0.1, 0.2)
) |>
  create_model_grid(
    f_model_spec = nnetar_reg,
    engine_name = "nnetar",
    mode = "regression",
    seasonal_period = 12
  )


# Build workflows ============================

arima_wfset <- workflow_set(
  preproc = list(
    base_rec = arima_rec1,
    econ_rec = arima_rec2,
    demog_rec = arima_rec3
  ),
  models = c(
    arima_default = list(arima_default),
    arima_spec = arima_spec_grid$.models
  ),
  cross = TRUE
)

ets_wfset <- workflow_set(
  preproc = list(base_rec = arima_rec1),
  models = c(ets_default = list(ets_default), ets_spec = ets_spec_grid$.models),
  cross = TRUE
)

tbats_wfset <- workflow_set(
  preproc = list(base_rec = arima_rec1),
  models = c(tbats_spec = tbats_spec_grid$.models),
  cross = TRUE
)

stlm_wfset <- workflow_set(
  preproc = list(base_rec = arima_rec1),
  models = c(stlm_spec = stlm_spec_grid$.models),
  cross = TRUE
)

pb_wfset <- workflow_set(
  preproc = list(pb_rec = pb_rec),
  models = c(pb_default = list(pb_default), pb_spec = pb_spec_grid$.models),
  cross = TRUE
)

nnet_wfset <- workflow_set(
  preproc = list(nnet_rec = nnet_rec),
  models = c(
    nnet_default = list(nnet_default),
    nnet_spec = nnet_spec_grid$.models
  ),
  cross = TRUE
)


# Combine all the workflows ===============

hpi_wfset <- bind_rows(
  arima_wfset, ets_wfset, tbats_wfset, stlm_wfset, pb_wfset, nnet_wfset
)


# Run all the models =================

num_cores <- sapply(2:15, \(x) nrow(hpi_wfset) %% x) |> which.max() + 1

# Start parallel processor
parallel_start(num_cores, .method = "parallel")

# Run all of the models
hpi_mods <- hpi_wfset |>
  modeltime_fit_workflowset(
    data = training(econ_splits_global),
    control = control_fit_workflowset(verbose = TRUE, allow_par = TRUE)
  )

# Get model accuracy figures
hpi_mod_calib <- hpi_mods |>
  modeltime_calibrate(testing(econ_splits_global), id = "city")

hpi_mod_accuracy <- hpi_mod_calib |>
  modeltime_accuracy() |>
  drop_na(.type)

hpi_local_mod_accuracy <- hpi_mod_calib |>
  modeltime_accuracy(acc_by_id = TRUE) |>
  drop_na(.type)

# Stop the parallel process
parallel_stop()


# Get failures ===============================

mod_fail_df <- hpi_mods |>
  mutate(mod_fail = map_lgl(.model, \(x) is.null(x))) |>
  select(!.model)

mod_fail_df |>
  group_by(mod_fail) |>
  slice_min(.model_id, n = 3)

failure_spec_df <- mod_fail_df |>
  filter(mod_fail) |>
  separate_wider_delim(
    .model_desc,
    delim = "_",
    names = c("rec_name", "rec", "algo", "spec")
  ) |>
  separate_wider_regex(spec, patterns = c(spec = "SPEC", spec_num = "\\d+"))

failure_spec_df |> distinct(rec_name, algo, spec)

failures <- failure_spec_df |>
  pull(spec_num) |>
  as.numeric()

ets_failed <- ets_spec_grid |>
  slice(failures) |>
  select(!.models)

ets_failed |>
  count(smooth_level, smooth_trend)

ets_failed |> count(smooth_seasonal)
ets_failed |> count(seasonal_period)

# Show that hpi_mod_accuracy and hpi_local_mod_accuracy dropped the failures
# Drop the failures from the calibration data

hpi_mod_calib <- hpi_mod_calib |>
  slice(which(!mod_fail_df$mod_fail))

# Best globals
best_global_table <- hpi_mod_accuracy |>
  slice_min(smape, n = 10)

best_local_table <- hpi_local_mod_accuracy |>
  group_by(city) |>
  slice_min(smape, n = 5)

best_global_table |>
  table_modeltime_accuracy(.interactive = FALSE)

best_local_table|>
  table_modeltime_accuracy(.interactive = FALSE)

best_mod_desc <- best_local_table |>
  pull(.model_desc) |>
  intersect(best_global_table$.model_desc) |>
  c(
    best_global_table |> slice_min(smape, n = 3) |> pull(.model_desc),
    best_local_table |>
      group_by(city) |>
      slice_min(smape, n = 1) |>
      pull(.model_desc)
  ) |>
  unique()

best_mod_desc <- best_mod_desc |>
  append(
    hpi_mod_accuracy |>
      filter(str_detect(.model_desc, "PB_|TBATS_")) |>
      separate_wider_delim(
        .model_desc,
        delim = "_",
        names = c("rec_name", "rec", "algo", "spec"),
        cols_remove = FALSE
      ) |>
      group_by(algo) |>
      slice_min(smape, n = 1) |>
      pull(.model_desc)
  ) |>
  sort()
