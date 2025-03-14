
library(remotes)


library(geodata)
getwd()
setwd("C:/Users/r/Desktop/rethink_resample")

models <- c("ACCESS-CM2")

scenarios <- c("126", "245", "370", "585")

variables <- c("tmin", "tmax", "prec", "bioc")

time_ranges <- c("2021-2040", "2041-2060", "2061-2080", "2081-2100")

base_download_path <- "CMIP6"

if (!dir.exists(base_download_path)) {
  dir.create(base_download_path)
}

download_data <- function(model, ssp, time_range) {
  ssp_folder <- file.path(base_download_path, model, paste0("ssp", ssp), time_range)
  dir.create(ssp_folder, recursive = TRUE, showWarnings = FALSE) 
  
  for (var in variables) {
    outf <- paste0("wc2.1_5m_", var, "_", model, "_ssp", ssp, "_", time_range, ".tif")
    durl <- paste0("https://geodata.ucdavis.edu/cmip6/5m/", model, "/ssp", ssp, "/", outf)
    poutf <- file.path(ssp_folder, outf)
    
    if (file.exists(poutf)) {

      message(paste("File already exists:", poutf))
    } else {
      response <- try(download.file(durl, poutf, mode = "wb"), silent = TRUE)
      if (inherits(response, "try-error")) {
        message(paste("Failed to download:", durl))
      } else {
        message(paste("Downloaded:", var, "under SSP", ssp, "for model", model, "for period", time_range))
      }
    }
  }
}

for (model in models) {
  for (ssp in scenarios) {
    for (time_range in time_ranges) {
      download_data(model, ssp, time_range)
    }
  }
}
