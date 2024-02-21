if (!"gt" %in% .packages()) suppressPackageStartupMessages(library("gt"))

gt_bold_head <- function(gt_tbl) {
  gt_tbl |>
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