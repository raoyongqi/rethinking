
library(lavaan)

library(semPlot)

getwd()
file_path = "data/selection.csv"

data <- read.csv(file_path)

library(dplyr)

data <- data %>%
  rename(Pathogen_Load = `pathogen.load`)
data <- data %>%
  rename(`hand` = hand_500m_china_03_08)

data <- data %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
data <- data %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)

if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  data <- data %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}

names(data)

library(rstanarm)
options(mc.cores = parallel::detectCores())
sem_model <- '
  Pathogen Load ~ lat + lon + Height + Richness

'
# 拟合 SEM 模型
fit <- sem(sem_model, data = data)

# 输出模型摘要
summary(fit, standardized = TRUE, fit.measures = TRUE)
# 生成路径图
semPaths(fit, 
         what = "std",        # 标准化路径系数
         layout = "circle",   # 布局
         edge.label.cex = 1.2, # 线条标签大小
         fade = FALSE,         # 保持线条颜色一致
         label.cex = 1.5,      # 节点标签大小

         residuals = FALSE)    # 不显示残差
