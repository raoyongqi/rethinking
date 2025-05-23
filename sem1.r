library(seminr)
getwd()


file_path = "data/selection.csv"

data <- read.csv(file_path)
library(dplyr)
# install.packages("sjSDM")
data <- data %>%
  rename(`Pathogen Load` = pathogen.load)

data <- data %>%
  rename(`hand` = hand_500m_china_03_08)

data <- data %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
data <- data %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)
colnames(data)



if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  df <- df %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}
# colnames(data) <- c(
#   "Latitude",                                       # lat
#   "Longitude",                                      # lon
#   "Area-weighted soil organic carbon content",      # awt_soc
#   "Dominant mapping unit ID",                       # dom_mu
#   "Soil Sand",                                      # s_sand
#   "Topsoil Sand",                                   # t_sand
#   "Height Above the Nearest Drainage",              # hand
#   "Solar Radiation",                                # srad
#   "Precipitation of Wettest Month",                 # bio_13
#   "Precipitation Seasonality",                      # bio_15
#   "Precipitation of Warmest Quarter",               # bio_18
#   "Precipitation of Coldest Quarter",               # bio_19
#   "Isothermality",                                  # bio_3
#   "Min Temperature of Coldest Month",               # bio_6
#   "Mean Temperature of Wettest Quarter",            # bio_8
#   "Wind Speed",                                     # wind
#   "Pathogen Load"                                   # Pathogen Load
# )
# 
# Climate <- c(
#   "Solar Radiation",                            # srad
#   "Precipitation of Wettest Month",             # bio_13
#   "Precipitation Seasonality",                  # bio_15
#   "Precipitation of Warmest Quarter",           # bio_18
#   "Precipitation of Coldest Quarter",           # bio_19
#   "Isothermality",                              # bio_3
#   "Min Temperature of Coldest Month",           # bio_6
#   "Mean Temperature of Wettest Quarter",        # bio_8
#   "Wind Speed"                                  # wind
# )
# 
# Soil <- c(
#   "Area-weighted Soil Organic Carbon Content",  # awt soc
#   "Dominant Mapping Unit ID",                   # dom mu
#   "Soil Sand",                                  # s_sand
#   "Topsoil Sand"                                # t_sand
# )
# 
# Geo <- c(
#   "Latitude",                                   # lat
#   "Longitude",                                  # lon
#   "Height Above the Nearest Drainage"           # hand
# )
columns <- c("lat", "lon", "awt_soc", "dom_mu", "s_sand", "t_sand", 
             "hand", "srad", "bio_13", "bio_15", "bio_18", "bio_19", 
             "bio_3", "bio_6", "bio_8", "wind")


Climate <- c("srad", "bio_13", "bio_15", "bio_18", "bio_19", 
             "bio_3", "bio_6", "bio_8", "wind")

Soil <- c("awt_soc", "dom_mu", "s_sand", "t_sand")

Geo <- c("lat", "lon", "hand")


measurements <- constructs(
  composite("Climate", Climate),  # 定义反射性构念 "Climate"
  composite("Soil", Soil),  # 定义复合构念 "Soil"
  composite("Geo", Geo),  # 定义复合构念 "Geo"
  composite("Pathogen Load",  single_item("Pathogen Load"))
)


str(multi_items("CUEX", 1:3)
)

structure <- relationships(
  paths(from = c("Climate", "Geo"), to = "Soil"),  # 气候和地理影响土壤
  
  paths(from = c("Climate", "Geo", "Soil"), to = "Pathogen Load")  # 气候、地理和土壤共同影响 Pathogen Load
)


thm <- seminr_theme_create(manifest.reflective.shape =  "ellipse",
                           manifest.compositeA.shape =  "hexagon",
                           manifest.compositeB.shape =  "box",
                           construct.reflective.shape = "hexagon",
                           construct.compositeA.shape = "box",
                           construct.compositeB.shape = "ellipse",
                           plot.rounding = 3, plot.adj = FALSE, 
                           sm.node.fill = "cadetblue1",
                           mm.edge.label.fontsize=14,
                           sm.edge.label.fontsize=15,

                           
                           sm.node.label.fontsize	=15,
                           mm.node.label.fontsize	=14,
                           mm.node.fill = "white")

# 估计 PLS 模型
pls_model <- estimate_pls(data = data, measurements, structure)
pls_model
boot_estimates <- bootstrap_model(pls_model, nboot = 1000, cores = 2)
# graph <- seminr::seminr_graph(pls_model, theme = thm)


boot_estimates


# layout_matrix <- layout_as_tree(graph)

seminr_theme_set(thm)
library(semPlot)
png("PLS_Model1.png", width = 12, height = 8, units = "in", res = 300)
plot(pls_model, edge.display = "values", loading.range = c(0.5, 1))  # 注意：仅在 semPlot 支持的对象上可用
dev.off()
theme_set(theme_minimal(base_family = "Arial Unicode MS"))
save_plot("myfigure.png")
dev.off()
library(ggplot2)
ggsave("PLS_Model.png", plot = last_plot(), dpi = 300, width = 12, height = 8)
