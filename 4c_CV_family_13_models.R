
getwd()


file_path = "data/selection.csv"

data <- read.csv(file_path)
# install.packages(c("caret", "Cubist", "randomForest", "xgboost", "glmnet", "pls", "kernlab", "penalized"))
library(caret)
packages <- c("caret", "e1071", "subselect", "ipred", "parallel", "doParallel", "corrplot",  "rowr", "ggplot2") 
sapply(packages, require, character.only = TRUE)
# 设置交叉验证参数
set.seed(7)
control <- trainControl(method = "cv", number = 10)  # 10折交叉验证
metric <- "RMSE" # 如果是分类任务，回归任务可以改为 "RMSE"

# 假设 df 是你的数据集，并且 "target" 是目标变量
target <- "pathogen.load"  # 替换成你的因变量名称

# 创建保存模型的文件夹
if (!dir.exists("data")) {
  dir.create("data")
}

# 训练多个模型并立即保存
train_and_save <- function(model_name, method, tuneGrid = NULL) {
  set.seed(7)
  # 训练模型
  model <- train(as.formula(paste(target, "~ .")), data = data, method = method, metric = metric, trControl = control, 
                 tuneGrid = tuneGrid, na.action = "na.exclude")
  
  # 保存模型
  save(model, file = paste0("data/", model_name, ".RData"))
  
  # 输出训练完成的消息
  print(paste(model_name, "训练完成并已保存"))
  
  # 返回训练好的模型，若需要进一步操作时可以使用
  return(model)
}

# 训练并保存多个模型
cubist_model <- train_and_save("cubist", "cubist", tuneGrid = data.frame(committees = 100, neighbors = 9))
rf_model <- train_and_save("rf", "rf", tuneGrid = expand.grid(.mtry = floor(sqrt(ncol(data[, !(colnames(data) %in% target)])))))
xgbLinear_model <- train_and_save("xgbLinear", "xgbLinear")
rqnc_model <- train_and_save("rqnc", "rqnc")
gamSpline_model <- train_and_save("gamSpline", "gamSpline")
penalized_model <- train_and_save("penalized", "penalized", tuneGrid = data.frame(lambda1 = 0, lambda2 = 1))
BstLm_model <- train_and_save("BstLm", "BstLm")
simpls_model <- train_and_save("simpls", "simpls")
widekernelpls_model <- train_and_save("widekernelpls", "widekernelpls")
glmnet_model <- train_and_save("glmnet", "glmnet")
gaussprPoly_model <- train_and_save("gaussprPoly", "gaussprPoly")
pcr_model <- train_and_save("pcr", "pcr")
lm_model <- train_and_save("lm", "lm")

cv_models_comp <- list(
  "cubist" = cubist_model,
  "rf" = rf_model,
  "xgbLinear" = xgbLinear_model,
  "rqnc" = rqnc_model,
  "gamSpline" = gamSpline_model,
  "penalized" = penalized_model,
  "BstLm" = BstLm_model,
  "simpls" = simpls_model,
  "widekernelpls" = widekernelpls_model,
  "glmnet" = glmnet_model,
  "gaussprPoly" = gaussprPoly_model,
  "pcr" = pcr_model,
  "lm" = lm_model
)
results <- resamples(cv_models_comp) 
summ <- summary(results)
dotplot(results)
print("所有模型已训练并保存到 data/ 文件夹！")
# 提取每个模型的 RMSE 或其他评估指标
# 提取每个模型的 RMSE 或其他评估指标
results <- data.frame(
  model = c("cubist_model", "rf_model", "xgbLinear_model", "rqnc_model", "gamSpline_model", 
            "penalized_model", "BstLm_model", "simpls_model", "widekernelpls_model", 
            "glmnet_model", "gaussprPoly_model", "pcr_model", "lm_model"),
  RMSE = c(cubist_model$results$RMSE,
           rf_model$results$RMSE,
           xgbLinear_model$results$RMSE,
           rqnc_model$results$RMSE,
           gamSpline_model$results$RMSE,
           penalized_model$results$RMSE,
           BstLm_model$results$RMSE,
           simpls_model$results$RMSE,
           widekernelpls_model$results$RMSE,
           glmnet_model$results$RMSE,
           gaussprPoly_model$results$RMSE,
           pcr_model$results$RMSE,
           lm_model$results$RMSE)
)
summary(lm_model)

# 查看结果
# summarize accuracy of models
results <- resamples(cv_models_comp) 
summ <- summary(results)
dotplot(results)
summ_stats
summ_stats <- do.call(rbind.data.frame, summ$statistics)
summ_stats$metric <- factor(gsub("(\\w+)\\.\\w+", "\\1", rownames(summ_stats)),
                            levels = c("Rsquared", "RMSE", "MAE"), labels = c("R-squared", "RMSE", "MAE"))     

summ_stats <- summ_stats[!(summ_stats$metric %in% c("MAE")), ]    
summ_stats <- rownames_to_column(summ_stats, var = "model")
summ_stats$model <- sub(".*\\.", "", summ_stats$model)


R2 <- summ_stats[with(summ_stats, metric %in% "R-squared"), ]
R2

summ_stats$model <- factor(summ_stats$model, levels = R2[order(R2$Mean, decreasing =FALSE), "model"])

str(summ_stats)
# 3. 获取排序后的 model 顺序
ordered_models <- R2_sorted

R2_sorted

library(tibble)

# Convert row names to a new 'model' column


# 4. 更新 summ_stats 的 model 列顺序
ggplot(summ_stats, aes(x = model, y = Mean)) +
  geom_linerange(aes(ymin = Min., ymax = Max.), size = 1.2) +  # 加粗线条
  geom_point(size = 3, shape = 16, color = "red") +  # 将点设置为红色
  
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x") +
  labs(x = "", y = "Predictive Accuracy") +
  theme_publication() +
  theme(
    axis.title.x = element_text(size = 14, margin = margin(t = 5, r = 0, b = 0, l = 0)),  # 增大字体
    axis.title.y = element_text(size = 14, margin = margin(t = 0, r = -3, b = 0, l = 0)), # 增大字体
    axis.text.x = element_text(size = 12),  # 增大字体
    axis.text.y = element_text(size = 12),  # 增大字体
    strip.text = element_text(size = 12),  # 增大字体
    panel.border = element_rect(size = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(size = .5, color = "#f0f0f0"),
    plot.margin = grid::unit(c(0, 0.1, 0.2, 0), units = "cm")
  )

 
 
 
 
 theme_publication <- function(base_size = 12, base_family = "Helvetica", ...) {
   require(grid)
   require(ggthemes)
   (theme_foundation(base_size = base_size, base_family = base_family)
     + theme(plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
             text = element_text(),
             panel.background = element_rect(color = NA),
             plot.background = element_rect(color = NA),
             panel.border = element_rect(color = "black", size = 1),
             axis.title = element_text(face = "plain", size = rel(1)),
             axis.title.y = element_text(angle=90, vjust = 2, margin = margin(r=7)),
             axis.title.x = element_text(vjust = -0.2, margin = margin(t=10)),
             axis.text = element_text(size = rel(0.9)), 
             #axis.line.y = element_line(color="black"),
             #axis.line.x = element_line(color="black"),
             axis.ticks = element_line(),
             panel.grid.minor = element_blank(),
             panel.grid.major.y = element_line(size=.5, color="#f0f0f0"),
             # explicitly set the horizontal lines (or they will disappear too)
             panel.grid.major.x = element_blank(),
             panel.spacing = unit(0.1, "lines"),
             #legend.key = element_rect(color = NA),
             legend.position = "none",
             #legend.direction = "horizontal",
             legend.key.size = unit(0.5, "cm"),
             legend.spacing = unit(0, "cm"),
             #legend.title = element_text(face="italic"),
             legend.text = element_text(size = 8),
             plot.margin = unit(c(10,5,5,5),"mm"),
             # strip.text = element_blank(),
             strip.background = element_blank()
     ))
 }
 