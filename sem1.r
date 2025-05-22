library(seminr)
getwd()


file_path = "data/selection.csv"

data <- read.csv(file_path)
library(dplyr)
install.packages("sjSDM")
data <- data %>%
  rename(`Pathogen Load` = pathogen.load)

data <- data %>%
  rename(`hand` = hand_500m_china_03_08)

data <- data %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
data <- data %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)

if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  df <- df %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}

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

colnames(data)

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
graph <- seminr::seminr_graph(pls_model, theme = thm)
boot_estimates
# 
# # 重新调整布局为从上到下
# layout_matrix <- layout_as_tree(graph)

# 绘制 PLS 模型
seminr_theme_set(thm)

png("PLS_Model1.png", width = 12, height = 8, units = "in", res = 300)  # 打开图形设备
plot(pls_model, main = "PLS Model")  # 绘制模型
dev.off()  # 关闭图形设备，保存文件
library(ggplot2)
ggsave("PLS_Model.png", plot = last_plot(), dpi = 300, width = 12, height = 8)
