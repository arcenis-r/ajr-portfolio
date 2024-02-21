
library(tidyverse)
library(timetk)
library(modeltime)
library(tidymodels)
library(gt)
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")
tidymodels_prefer()


# Import and split data ========================================================

econ_data <- read_rds("./posts/_common-resources/econ-data.Rds")

econ_splits <- econ_data |>
  time_series_split(
    assess = "2 year",
    cumulative = TRUE,
    date_var = date
  )

# Write recipes ================================================================

arima_rec1 <- recipe(hpi ~ date, data = training(econ_splits))

arima_rec2 <- recipe(
  hpi ~ date + unemp_rate,
  data = training(econ_splits)
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
  data = training(econ_splits)
) |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>
  step_lag(unemp_rate, population, lag = c(1, 3, 6))

nnet_rec <- recipe(
  hpi ~ .,
  data = training(econ_splits)
) |>
  update_role(city, new_role = "ID") |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>
  step_timeseries_signature(date) |>
  step_nzv(all_predictors())


# Specify models ===============================================================

arima_default <- arima_reg() |> set_engine("auto_arima")                        # <1>

arima_spec_grid <- tibble(seasonal_period = seq(9, 18, by = 3)) |>              # <2>
  create_model_grid(                                                            # <3>
    f_model_spec = arima_reg,
    engine_name = "auto_arima",
    mode = "regression"
  )

ets_default <- exp_smoothing() |> set_engine("ets")

set.seed(40)                                                                    # <4>
alpha_vals <- runif(5, 0.01, 0.5) *
  sample(c(1, 0.1, 0.01, .001), 5, replace = TRUE)
beta_vals <- runif(5, 0.01, 0.2) * sample(c(1, 0.1, 0.01), 5, replace = TRUE)
gamma_vals <- map_dbl(
  alpha_vals,
  \(x) runif(1, 0, 1 - x)
)

ets_spec_grid <- expand_grid(                                                   # <5>
  smooth_level = alpha_vals[-2],
  smooth_seasonal = gamma_vals[2:3],
  seasonal_period = seq(9, 18, by = 3)
) |>
  create_model_grid(
    f_model_spec = exp_smoothing,
    engine_name = "ets",
    mode = "regression",
    error = "auto",
    trend = "auto",
    season = "auto",
    smooth_trend = beta_vals[3]
  )

stlm_spec_grid <- expand_grid(
  seasonal_period_2 = c(NULL, seq(6, 18, by = 3))
) |>
  create_model_grid(
    f_model_spec = seasonal_reg,
    engine_name = "stlm_arima",
    mode = "regression",
    seasonal_period_1 = 12
  )

nnet_default <- nnetar_reg() |> set_engine("nnetar")

nnet_spec_grid <- expand_grid(
  non_seasonal_ar = 1:3,
  hidden_units = 3:5,
  num_networks = c(20, 25),
  penalty = c(0.2, 0.25)
) |>
  create_model_grid(
    f_model_spec = nnetar_reg,
    engine_name = "nnetar",
    mode = "regression",
    seasonal_period = 12,
    epochs = 100,
    seasonal_ar = 1
  )


# Build workflowsets ===========================================================

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

stlm_wfset <- workflow_set(
  preproc = list(base_rec = arima_rec1),
  models = c(stlm_spec = stlm_spec_grid$.models),
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

hpi_wfset <- bind_rows(arima_wfset, ets_wfset, stlm_wfset, nnet_wfset)