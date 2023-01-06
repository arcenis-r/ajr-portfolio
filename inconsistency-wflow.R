library(cepumd)
library(readxl)
library(tidyverse)

data_dir <- file.path(tempdir(), "ce-pet-data")
dir.create(data_dir)

store_ce_dict(data_dir, "ce-data-dictionary.xlsx")
store_ce_hg(data_dir, "ce-stubs.zip")
ce_download(2015, interview, data_dir)
ce_download(2015, diary, data_dir)

list.files(data_dir)


# Check codes for FAM_TYPE
ce_codes <- read_excel(
  file.path(data_dir, "ce-data-dictionary.xlsx"),
  sheet = "Codes "
) %>%
  janitor::clean_names() %>%
  select(survey:last_quarter)

ce_codes %>% filter(
  variable %in% "FAM_TYPE",
  first_year <= 2015,
  (last_year >= 2015 | is.na(last_year)),
  code_value %in% c("3", "5", "7")
) %>%
  select(survey, code_value, code_description) %>%
  arrange(code_value, survey)

# Create my own dictionary (keep Interview code definitions)
fam_type_codes <- ce_codes %>%
  filter(
    variable %in% "FAM_TYPE",
    first_year <= 2015,
    (last_year >= 2015 | is.na(last_year))
  )

codes2keep <- fam_type_codes %>%
  filter(survey %in% "INTERVIEW") %>%
  select(code_value, code_description)

fam_type_codes <- fam_type_codes %>%
  select(-code_description) %>%
  left_join(codes2keep, by = "code_value") %>%
  # mutate(survey = str_sub(survey, 1, 1), variable = str_to_lower(variable)) %>%
  relocate(code_description, .after = code_value)

fam_type_codes %>%
  filter(code_value %in% c("3", "5", "7")) %>%
  select(survey, code_value, code_description) %>%
  arrange(code_value, survey)


hg_15 <- ce_hg(2015, integrated, data_dir, "ce-stubs.zip")
ce_prepdata(
  2015,
  integrated,
  ce_uccs(hg_15, expenditure = "Pets", ucc_group = "PETS", uccs_only = TRUE),
  recode_variables = TRUE,
  own_codebook = fam_type_codes,
  ce_dir = data_dir,
  int_zp = "intrvw15.zip",
  dia_zp = "diary15.zip",
  hg = hg_15,
  fam_type
) %>%
  nest(data = -fam_type) %>%
  mutate(ce_mean_df = map(data, ce_mean)) %>%
  select(-data) %>%
  unnest(ce_mean_df) %>%
  arrange(fam_type)
