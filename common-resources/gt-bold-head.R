if (!"gt" %in% .packages()) suppressPackageStartupMessages(library("gt"))

gt_bold_head <- function(df, grp_col = NULL) {
  df |>
    gt(groupname_col = grp_col) |>
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_column_labels()
    ) |>
    tab_style(
      style = cell_borders(
        sides = c("top", "bottom"),
        weight = px(2),
        style = ("solid")
      ),
      locations = cells_column_labels()
    )
}