library(seminr)
getwd()
# 读取数据
file_path = "data/selection.csv"  # 替换为你的文件路径
data <- read.csv(file_path)
library(dplyr)
# 假设你的数据框是 data
data <- data %>%
  rename(`Pathogen Load` = pathogen.load)
colnames(data)
data <- data %>%
  rename(`hand` = hand_500m_china_03_08)
data <- data %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
data <- data %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)
data <- data %>%
  rename(pct_clay = hwsd_soil_clm_res_pct_clay)
Climate <- c("srad", "bio_15", "wind", "bio_18", "bio_11", "bio_3", "bio_8")
Soil <- c("s_sand", "t_sand", "awt_soc", "pct_clay", "dom_mu")
Geo <- c("lat", "lon", "hand")


# 创建测量模型（反射性和复合指标）
measurements <- constructs(
  composite("Climate", Climate),  # 定义反射性构念 "Climate"
  composite("Soil", Soil),  # 定义复合构念 "Soil"
  composite("Geo", Geo),  # 定义复合构念 "Geo"
  composite("Pathogen Load",  single_item("Pathogen Load"))
)
str(Geo)
str(multi_items("CUEX", 1:3)
)

# 结构关系
structure <- relationships(
  paths(from = c("Climate", "Geo"), to = "Soil"),  # 气候和地理影响土壤
  
  paths(from = c("Climate", "Geo", "Soil"), to = "Pathogen Load")  # 气候、地理和土壤共同影响 Pathogen Load
)
# 检查数据类型
colnames(data)
# 如果需要，转换某些列为数值型
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
# graph <- seminr::seminr_graph(pls_model, theme = thm)
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
