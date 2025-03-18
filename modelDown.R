# Prepare the data
# 读取数据

library(dplyr)
file_path <- "data/selection.csv"  # 替换为你的文件路径
# Random Forest model
library(DALEX)
df <- read.csv(file_path)
colnames(df)
# 获取所有列名
df <- df %>%
  rename(`hand` = hand_500m_china_03_08)
df <- df %>%
  rename(`dom_mu` = hwsd_soil_clm_res_dom_mu)
df <- df %>%
  rename(`awt_soc` = hwsd_soil_clm_res_awt_soc)
if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  df <- df %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}

library("randomForest")

model_pl_rf <- randomForest(pathogen.load ~ .,  data = df)

explainer_rf <- DALEX::explain(model_pl_rf, 
                        data = df[, -which(names(df) == "pathogen.load")],  # 移除响应变量
                        y = df$pathogen.load, 
                        label = "Random Forest v7")

library("e1071")

model_pl_svm <- svm(pathogen.load ~ .,  data = df,
                         type = "eps-regression", probability = TRUE)
explain_pl_svm <- DALEX::explain(model_pl_svm, data = df, 
                               y = df$pathogen.load, 
                               label = "Support Vector Machines")

library("gbm")

model_pl_gbm <- gbm(
  pathogen.load ~ ., 
  data =df, 
  n.trees = 15000
)


explain_pl_gbm <- DALEX::explain(model_pl_gbm, data = df, 
                               y = df$pathogen.load, 
                               predict_function = function(m,x) predict(m, x, n.trees = 15000, type = "response"),
                               label = "Generalized Boosted Models")

# k-NN model

library("caret")
model_pl_knn <- knnreg(pathogen.load ~ .,data = df, k = 5)
explain_pl_knn <- DALEX::explain(model_pl_knn, data = df, 
                               y = df$pathogen.load, 
                               predict_function = function(m,x) predict(m, x),
                               label = "k-Nearest Neighbours")
# 加载所需的库
library(psych)
library(openxlsx)

# 使用 psych::describe 生成描述性统计信息
summary_df <- psych::describe(df)
# 保存为 Excel 文件

summary_df$vars <- rownames(summary_df)

summary_df$min <- as.numeric(summary_df$min)
summary_df$max <- as.numeric(summary_df$max)

# 将 row.names 转换为一个新的列 'vars'，并移除原有的 row.names
summary_df <- summary_df %>% select(vars, everything())  # 将 vars 列放到最前面

summary_df$range <- paste0(
  sprintf("%.2f", summary_df$min), 
  " - ", 
  sprintf("%.2f", summary_df$max)
)
summary_df <- as.data.frame(summary_df)

summary_df$mean <- round(summary_df$mean, 2)
summary_df$sd <- round(summary_df$sd, 2)
summary_df$skew <- round(summary_df$skew, 2)
summary_df$kurtosis <- round(summary_df$kurtosis, 2)
summary_df$se <- round(summary_df$se, 2)

units <- list(
  'lat' = '°',
  'lon' = '°',
  'awt_soc' = 'kg C m-2',
  'dom_mu' = 'numerical ID',
  'pct_clay' = '%',
  's_sand' = '%',
  't_sand' = '%',
  'hand' = 'm',
  'srad' = 'kJ m-2 day-1',
  'bio_11'= '°C',
  'bio_13' = 'mm',
  'bio_15' = 'mm',
 'bio_18' = 'mm',
  'bio_3' = '%',
  'bio_8' = '°C',
  'wind' = 'm/s'
)
# 定义 simplified_names 字典
simplified_names <- list(
  'srad' = 'Solar Radiation',
  's_sand' = 'Soil Sand',
  'lon' = 'Longitude',
  'wind' = 'Wind Speed',
  'lat' = 'Latitude',
  'ELEV' = 'Elevation',
  'MAT' = 'MeanAnnual Temperature',
  'MAP' = 'MeanAnnual Precipitation',
  't_sand' = 'Topsoil Sand',    
  'bio_1' = 'Annual Mean Temperature',
  'bio_2' = 'Mean Diurnal Range （Mean of monthly （max temp - min temp））',
  'bio_3' = 'Isothermality （bio_2/bio_7） （×100）',
  'bio_4' = 'Temperature Seasonality （standard deviation ×100）',
  'bio_5' = 'Max Temperature of Warmest Month',
  'bio_6' = 'Min Temperature of Coldest Month',
  'bio_7' = 'Temperature Annual Range （bio_5-bio_6）',
  'bio_8' = 'Mean Temperature of Wettest Quarter',
  'bio_9' = 'Mean Temperature of Driest Quarter',
  'bio_10' = 'Mean Temperature of Warmest Quarter',
  'bio_11' = 'Mean Temperature of Coldest Quarter',
  'bio_12' = 'Annual Precipitation',
  'bio_13' = 'Precipitation of Wettest Month',
  'bio_14' = 'Precipitation of Driest Month',
  'bio_15' = 'Precipitation Seasonality （Coefficient of Variation）',
  'bio_16' = 'Precipitation of Wettest Quarter',
  'bio_17' = 'Precipitation of Driest Quarter',
  'bio_18' = 'Precipitation of Warmest Quarter',
  'bio_19' = 'Precipitation of Coldest Quarter',
  'elev' = 'Elevation',
  'awt_soc' = 'Area-weighted soil organic carbon content',
  'dom_mu' = 'Dominant mapping unit ID from HWSD （MU_GLOBAL above）',
  'pct_clay' = 'Soil clay fraction by percent weight',
  's_clay' = 'Soil Clay',
  's_sand' = 'Soil Sand',
  't_clay' = 'Topsoil Clay',
  't_sand' = 'Topsoil Sand',
  'hand' = 'Height Above the Nearest Drainage'
)

# 假设 summary_df 是您的数据框
summary_df$simplified_names <- sapply(summary_df$vars, function(x) simplified_names[[x]])

# 向 summary_df 添加一个新列，显示变量和单位
summary_df$variable_with_unit <- paste0(summary_df$vars, "（", sapply(summary_df$vars, function(x) units[[x]]), "）")

summary_df <- summary_df[, c( "variable_with_unit","simplified_names", setdiff(names(summary_df), c("simplified_names", "variable_with_unit")))]

summary_df <- summary_df[, c("variable_with_unit","simplified_names", "n", "mean", "sd", "skew", "kurtosis", "se", "range")]



summary_df
summary_df <- summary_df %>%
  rename(`var` =variable_with_unit)
# 将结果保存为 Excel 文件，并不保存行索引
write.xlsx(summary_df, "summary_stats.xlsx", rowNames = FALSE)

# Website generation
modelDown(explain_pl_rf, explain_pl_gbm, 
          explain_pl_svm, explain_pl_knn,
          device = "svg",
          remote_repository_path = "MI2DataLab/modelDown_example/docs",
          output_folder = "modelDown_pl_example")
