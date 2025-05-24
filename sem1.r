library(seminr)
getwd()

rm(list = ls())
file_path = "data/selection.csv"

data <- read.csv(file_path)
library(dplyr)
# install.packages("sjSDM")
data <- data %>%
  rename(`PathogenLoad` = pathogen.load)

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
colnames(data) <- c(
  "Latitude",                                   # lat
  "Longitude",                                  # lon
  "Area_weighted_Soil_Organic_Carbon_Content",  # awt soc
  "Dominant_Mapping_Unit_ID",                   # dom mu
  "Soil_Sand",                                  # s_sand
  "Topsoil_Sand",                               # t_sand
  "Height_Above_the_Nearest_Drainage",          # hand
  "Solar_Radiation",                            # srad
  "Precipitation_of_Wettest_Month",             # bio_13
  "Precipitation_Seasonality",                  # bio_15
  "Precipitation_of_Warmest_Quarter",           # bio_18
  "Precipitation_of_Coldest_Quarter",           # bio_19
  "Isothermality",                              # bio_3
  "Min_Temperature_of_Coldest_Month",           # bio_6
  "Mean_Temperature_of_Wettest_Quarter",        # bio_8
  "Wind_Speed",                                 # wind
  "Pathogen_Load"                               # Pathogen Load
)

Climate <- c("Solar_Radiation", "Precipitation_of_Wettest_Month", "Precipitation_Seasonality", 
             "Precipitation_of_Warmest_Quarter", "Precipitation_of_Coldest_Quarter", 
             "Isothermality", "Min_Temperature_of_Coldest_Month", "Mean_Temperature_of_Wettest_Quarter", "Wind_Speed")


Soil <- c("Area_weighted_Soil_Organic_Carbon_Content", "Dominant_Mapping_Unit_ID", "Soil_Sand", "Topsoil_Sand")

Geography <- c("Latitude", "Longitude", "Height_Above_the_Nearest_Drainage")


measurements <- constructs(
  composite("Climate", Climate),  # 定义反射性构念 "Climate"
  composite("Soil", Soil),  # 定义复合构念 "Soil"
  composite("Geography", Geography),  # 定义复合构念 "Geo"
  reflective("Pathogen_Load",  single_item("Pathogen_Load"))
)
structure <- relationships(
  paths(to = "Pathogen_Load",
        from = c("Climate", "Geography", "Soil","Geography*Soil"))
)

source("C:/Users/r/Desktop/rethink_resample/seminrvis.R")

my_theme <- create_theme(item_style = createItemStyle(
    fontsize = 20, height = 0.6, width = 8,fill = "lightgoldenrodyellow"),

    # we can style the construct appearance
    construct_style = createConstructStyle(
      fontsize = 20, height = 1.6, width = 2,fill = "lightcyan"),
    outer_weight_style = createOuterWeightStyle(color = "blue", fontsize = 20),
    
    inner_weight_style = createInnerWeightStyle(color = "black", fontsize = 20)
)

structure
pls_model <- estimate_pls(data = data, measurements, structure)
plot_object
pls_model
plot_object <- pls_model %>% 
  plot_model(title = "", theme = my_theme) %>% 
  grViz()
plot_object <- pls_model %>% 
  plot_model(title = "", theme = my_theme)
plot_object
# 保存为PNG的可靠方法
tmp_svg <- tempfile(fileext = ".svg")
plot_object %>% 
  DiagrammeRsvg::export_svg() %>% 
  writeLines(tmp_svg)

rsvg::rsvg_png(
  tmp_svg, 
  "model_plot.png", 
  width = 1200, 
  height = 1200
)
plot_object
# graph <- seminr::seminr_graph(pls_model, theme = thm)
# 
# boot_estimates <- bootstrap_model(pls_model, nboot = 1000, cores = 2)
# 
# boot_estimates



library(ggplot2)


install.packages("C:/Users/r/Downloads/seminr-vis.zip", 
                 repos = NULL, 
                 type = "win.binary",
                 dependencies = TRUE)
str(boot_estimates)
plot(boot_estimates, plottype = "boxplot", 
     labels = rownames(boot_estimates$coefficients))

# layout_matrix <- layout_as_tree(graph)
install.packages("ggbiplot")
my_plot <- recordPlot()
library(devtools)
install_github('zdk123/compPLS')
boot_data <- as.data.frame(boot_estimates$bootCoeffs)
boot_data_long <- stack(boot_data)
boot_data <- as.data.frame(boot_estimates$bootCoeffs)
class(boot_estimates$bootCoeffs)
dim(boot_estimates$bootCoeffs)
boot_data_long <- stack(boot_data)
# 2. 创建ggplot对象
library(ggplot2)
gg_boot <- ggplot(boot_data_long, aes(x = ind, y = values)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Bootstrap系数分布",
       x = "预测变量",
       y = "系数值") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

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
