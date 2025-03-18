# 加载必要的库
rm(list = ls())
library(vegan)

library(linkET)
library(ggplot2)
getwd()
# 读取数据
file_path = "data/selection.csv"  # 替换为你的文件路径
data <- read.csv(file_path)
library(dplyr)
# 假设你的数据框是 data
data <- data %>%
  rename(`Pathogen Load` = pathogen.load)
# 获取所有列名
data <- data %>%
  rename(`hand` = hand_500m_china_03_08)
data <- data %>%
  rename(`dom mu` = hwsd_soil_clm_res_dom_mu)
data <- data %>%
  rename(`awt soc` = hwsd_soil_clm_res_awt_soc)
if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  df <- df %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}

other_columns <- setdiff(names(data), "Pathogen Load")
ncol(data)
colnames(data)
# 使用其他列名从数据框中选取这些列
xxxx <- data[, other_columns]

# 提取因变量
yyyy <- data$`Pathogen Load`

Climate <- c("srad", "bio_15", "wind", "bio_18", "bio_11", "bio_3", "bio_8")
Soil <- c("s_sand", "t_sand", "awt soc", "pct clay", "dom mu")
Geo <- c("lat", "lon", "hand")

# 假设 df 是你原来的数据框
# 根据列列表重新排序
xxxx <- xxxx %>%
  select(all_of(c(Climate, Soil, Geo)))

# 使用 mantel_test 进行 Mantel 检验
mantel <- mantel_test(yyyy, xxxx,
                      spec_select = list(`Pathogen Load` = 1:1)
)%>% 
  mutate(rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf),
                  labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
         pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf),
                  labels = c("< 0.01", "0.01 - 0.05", ">= 0.05")))
mantel_subset <- mantel[, 1:4]
# 加载 openxlsx 包
library(openxlsx)

# 提取前 4 列
mantel_subset <- mantel[, 1:4]

# 创建 Excel 工作簿
wb <- createWorkbook()
addWorksheet(wb, "Mantel Results")

# 写入数据
writeData(wb, "Mantel Results", mantel_subset)

# 识别显著性 p 值（例如 p < 0.05）
sig_indices <- which(mantel_subset$p < 0.05, arr.ind = TRUE)

# 设置加粗格式
bold_style <- createStyle(textDecoration = "bold")

# 应用加粗格式
for (i in sig_indices) {
  addStyle(wb, "Mantel Results", bold_style, rows = i + 1, cols = 4, gridExpand = TRUE)
}
saveWorkbook(wb, "mantel_results.xlsx", overwrite = TRUE)

qcorrplot(correlate(xxxx)
, type = "lower", diag = FALSE)+
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), 
              data = mantel, 
              curvature = nice_curvature()) +
  scale_fill_gradientn(colours = RColorBrewer::brewer.pal(11, "RdBu")) +
  scale_size_manual(values = c(0.5, 1, 2)) +
  scale_colour_manual(values = color_pal(3)) +
  guides(size = guide_legend(title = "Mantel's r",
                             override.aes = list(colour = "grey35"), 
                             order = 2),
         colour = guide_legend(title = "Mantel's p", 
                               override.aes = list(size = 3), 
                               order = 1),
         fill = guide_colorbar(title = "Pearson's r", order = 3))

traceback()
# 在 R 启动时增加 C stack 的大小
options(expressions = 500000)  # 增加最大表达式数限制
