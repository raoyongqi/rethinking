setwd("C:\\Users\\r\\Downloads\\r-lille-2021-9d65898d91ce9c62967c6cea44605375a9f41b8d\\r-lille-2021-9d65898d91ce9c62967c6cea44605375a9f41b8d\\churn")

source("R\\functions.R")
tar_option_set(packages = c("keras", "tidyverse", "rsample", "recipes", "yardstick"))
library(reticulate)
devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow()

# virtualenv_create('r-reticulate')
py_config()
Sys.setenv("RETICULATE_PYTHON" = "C:/Python/Python311/python.exe")

tensorflow::install_tensorflow(
  version = "release", # or "2.16" or "2.17" 
  envname = "r-tensorflow", 
  extra_packages = "tf_keras", # legacy keras
  python_version = "3.11"
)
library(targets)



list(
  tar_target(file, "data/churn.csv", format = "file"),
  tar_target(data, split_data(file)),
  tar_target(recipe, prepare_recipe(data)),
  tar_target(model1, test_model(data, recipe, act1 = "relu")),
  tar_target(model2, test_model(data, recipe, act1 = "sigmoid")),
  tar_target(model3, test_model(data, recipe, act1 = "linear"))
)
tar_visnetwork(targets_only = TRUE)
tar_make()
conda_create()
tar_visnetwork(targets_only = TRUE)
