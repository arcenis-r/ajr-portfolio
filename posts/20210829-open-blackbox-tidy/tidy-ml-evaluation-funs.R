# Create a ggplot2 theme to use throughout the project
ml_eval_theme <- function() {
  theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18),
      strip.text = element_text(size = 12, color = "white"),
      strip.background = element_rect(fill = "#17468F")
    )
}

# Reverse the Yeo-Johnson transformation
# Note: This will only be accurate if the original data was all greater than 0
#   and lambda was not equal to 0. In the case of home prices the first is
#   true.
rev_yj <- function(y_tf, lam) {
  if (lam == 0) {
    exp(y_tf) - 1
  } else {
    (((y_tf * lam) + 1)^(1 / lam)) - 1
  }
}

# Prediction wrapper to be used with vip::vi_permute
tidy_pred <- function(object, newdata) {
  predict.model_fit(object, new_data = newdata, type = "numeric") |>
    pull(.pred)
}

# Calculate permutation variable importance
get_var_imp <- function(wflow) {
  mod <- pull_workflow_fit(wflow)
  train_dat <- pull_workflow_mold(wflow) |> pluck("predictors")
  target_dat <- pull_workflow_mold(wflow) |>
    pluck("outcomes") |>
    pull(sale_price)

  vi_df <- vip::vi_permute(
    mod,
    train = train_dat,
    target = target_dat,
    metric = "mae",
    pred_wrapper = tidy_pred,
    nsim = 10,
    paralell = TRUE
  )
}

# Get a matrix of SHAP values for each variable for each observation
get_shap <- function(wflow) {
  fastshap::explain(
    extract_fit_parsnip(wflow),
    X = extract_mold(wflow) |> pluck("predictors") |> as.matrix(),
    feature_names = extract_mold(wflow) |>
      pluck("predictors") |>
      colnames(),
    pred_wrapper = tidy_pred,
    nsim = 10
  )
}

# Calculate SHAP variable importance
get_shap_imp <- function(shap_obj) {
  shap_obj |>
    as_tibble() |>
    summarise(across(everything(), \(x) mean(abs(x)))) |>
    pivot_longer(
      everything(),
      names_to = "Variable",
      values_to = "Importance"
    ) |>
    arrange(desc(Importance))
}

# Make dataframes with variable names, SHAP values (for each observation),
# feature values (for each observation), and variable importance
get_shap_summary <- function(vi, shap_df, feat_df, max_features = 20) {
  vi <- vi %>%
    set_names(colnames(.) |> str_to_lower()) |>
    slice_max(importance, n = max_features, with_ties = FALSE)

  shap_df <- shap_df |>
    as_tibble() |>
    mutate(id = row_number()) |>
    pivot_longer(cols = -id, names_to = "variable", values_to = "shap_value") |>
    mutate(shap_value = as.numeric(shap_value))

  feat_df <- feat_df |>
    mutate(id = row_number()) |>
    pivot_longer(cols = -id, names_to = "variable", values_to = "feature_value")

  left_join(shap_df, feat_df, by = c("variable", "id")) |>
    right_join(vi, by = "variable") |>
    mutate(
      variable = str_glue(
        "{variable} ({format(round(importance, 3), nsmall = 3)})"
      )
    ) |>
    arrange(desc(importance))
}

# Generate a variable importance plot
plot_var_imp <- function(vi_df, algorithm, max_features = 20) {
  vi_plot_df <- vi_df %>%
    slice_max(Importance, n = max_features) %>%
    arrange(desc(Importance))

  vi_plot_df %>%
    vip::vip() +
    labs(title = str_glue("Variable Importance: {algorithm}")) +
    ml_eval_theme()
}

# Generate a SHAP summary plot
plot_shap_summary <- function(shap_summary_df, algorithm) {
  shap_summary_df |>
    ggplot(
      aes(shap_value, fct_rev(fct_inorder(variable)), color = feature_value)
    ) +
    geom_jitter(height = 0.2, alpha = 0.8) +
    geom_vline(xintercept = 0, linewidth = 1) +
    scale_color_gradient(low = "blue", high = "red") +
    labs(
      title = str_glue("SHAP Summary Plot: {algorithm}"),
      x = "SHAP Value",
      y = "Variable (SHAP Importance)",
      color = "Feature Value"
    ) +
    ml_eval_theme()
}

# Generate a partial dependence plot
plot_pdp <- function(wflow, pred_var, algorithm) {
  pred_var <- enquo(pred_var)
  mod <- extract_fit_parsnip(wflow)
  train_dat <- extract_mold(wflow) |> pluck("predictors")

  pdp::partial(
    mod,
    pred.var = quo_name(pred_var),
    train = train_dat,
    type = "regression"
  ) |>
    data.frame() |>
    ggplot(aes(!!pred_var, yhat)) +
    geom_line() +
    stat_smooth(method = "loess", formula = y ~ x) +
    ggtitle(
      str_glue(
        "Partial Dependence of sale_price on {quo_name(pred_var)} ({algorithm})"
      )
    ) +
    ml_eval_theme()
}

# Generate a dataframe of top 'nfeat' SHAP contributions for a given observation
# and accompanying contribution plot
get_contributions <- function(shap_df, algorithm, rnum, nfeat) {
  contrib_df <- shap_df |>
    as_tibble() |>
    mutate(across(everything(), as.numeric)) |>
    dplyr::slice(rnum) |>
    pivot_longer(everything(), names_to = "variable", values_to = "shap")

  contrib_plot <- contrib_df |>
    slice_max(abs(shap), n = nfeat, with_ties = FALSE) |>
    ggplot(aes(x = shap, y = fct_reorder(variable, abs(shap)), fill = shap)) +
    geom_col(show.legend = FALSE) +
    labs(
      y = NULL,
      x = "Shapley value",
      title = str_glue(
        "Top {nfeat} SHAP Contributions for Observation {rnum} ({algorithm})"
      )
    ) +
    ml_eval_theme()

  return(list(contrib_df = contrib_df, contrib_plot = contrib_plot))
}
