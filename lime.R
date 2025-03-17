# 加载所需的库
library(DALEX)
library(randomForest)
# 加载必要的库
library(vegan)
library(linkET)
library(corrr)
library(ggplot2)
library(dplyr)
library(randomForest)


# 读取数据
file_path <- "data/selection.csv"  # 替换为你的文件路径
df <- read.csv(file_path)
getwd()
# 删除 'RATIO' 列
df <- df %>%
  select(pathogen.load, everything())
# 创建一个模型

df$lat <- zoo::na.locf(df$lat)  # 用前一个非缺失值填充
df <- df %>%
  rename(`hand` = hand_500m_china_03_08)
df <- df %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
df <- df %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)
df <- df %>%
  rename(pct_clay = hwsd_soil_clm_res_pct_clay)
model <- randomForest(pathogen.load ~ ., data = df)

# 创建一个 DALEX 解释器
explainer <- DALEX::explain(model, data = df[,-1], y = df$pathogen.load)

# 选择一个观察对象进行局部解释
explanation <- DALEX::predict_parts(explainer, new_observation = df[1,-1])
explanation
# 可视化局部解释，类似于 LIME 图
p <- plot(explanation) + 
  theme(
    text = element_text(size = 16),        # 设置字体大小为16
    axis.title = element_text(size = 18),  # 设置坐标轴标题字体大小
    axis.text = element_text(size = 14),   # 设置坐标轴刻度标签字体大小
    plot.title = element_text(size = 20)   # 设置图表标题字体大小
  )

# 保存图像
ggsave("lime.png", plot = p, dpi = 300, width = 12, height = 8)

