library("bayesplot")
library("rstanarm")
library("ggplot2")
library("xtable")
library(dplyr)

file_path <- "data/selection.csv"  # 替换为你的文件路径
selection <- read.csv(file_path)
# 获取所有列名
selection <- selection %>%
  rename(`hand` = hand_500m_china_03_08)
selection <- selection %>%
  rename(dom_mu = hwsd_soil_clm_res_dom_mu)
selection <- selection %>%
  rename(awt_soc = hwsd_soil_clm_res_awt_soc)
if ("hwsd_soil_clm_res_pct_clay" %in% colnames(df)) {
  df <- df %>%
    rename(`pct_clay` = hwsd_soil_clm_res_pct_clay)
}

selection_scaled <- selection
colnames(selection_scaled)
colnames(selection_scaled) <- c(
  "Latitude",                                                # lat
  "Longitude",                                               # lon
  "Area-weighted soil organic carbon content",               # awt_soc
  "Dominant mapping unit ID",    # dom_mu
  "Soil Sand",                                               # s_sand
  "Topsoil Sand",                                            # t_sand
  "Height Above the Nearest Drainage",                       # hand
  "Solar Radiation",                                         # srad
  "Precipitation of Wettest Month",                          # bio_13
  "Precipitation Seasonality",    # bio_15
  "Precipitation of Warmest Quarter",                        # bio_18
  "Precipitation of Coldest Quarter",                        # bio_19
  "Isothermality",                      # bio_3
  "Min Temperature of Coldest Month",                        # bio_6
  "Mean Temperature of Wettest Quarter",                     # bio_8
  "Wind Speed",                                              # wind
  "pathogen.load"                                            # pathogen.load
)

selection_scaled[ , -which(names(selection) == "pathogen.load")] <- scale(selection[ , -which(names(selection) == "pathogen.load")])

fit <- stan_glm(pathogen.load ~ ., data = selection_scaled)
posterior <- as.matrix(fit)
params <- colnames(posterior)

exclude_params <- c("(Intercept)", "sigma")

params_to_plot <- setdiff(params, exclude_params)

library(ggplot2)
library(bayesplot)


plot_title <- ggtitle("Posterior Distributions of Regression Coefficients",
                      "with medians and 80% intervals")
posterior
plot <- mcmc_areas_data(posterior,
                   pars = params_to_plot,
                   prob = 0.8) 
theme_publication <- function(base_size = 12, base_family = "Helvetica", ...) {
  require(grid)
  require(ggthemes)
  (theme_foundation(base_size = base_size, base_family = base_family)
    + theme(plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(color = NA),
            plot.background = element_rect(color = NA),
            panel.border = element_rect(color = "black", size = 1),
            axis.title = element_blank(),  # 隐藏 x 和 y 轴标题
            
            axis.text = element_text(size = rel(0.9)), 
            #axis.line.y = element_line(color="black"),
            #axis.line.x = element_line(color="black"),
            axis.ticks = element_line(),
            legend.position = "none",
            
            panel.grid.minor = element_blank(),
            panel.grid.major.y = element_line(size=.5, color="#f0f0f0"),
            # explicitly set the horizontal lines (or they will disappear too)
            panel.grid.major.x = element_blank(),
            panel.spacing = unit(0.1, "lines"),
            #legend.key = element_rect(color = NA),
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


library(ggridges)
plot$parameter <- factor(plot$parameter, levels = unique(plot$parameter))  # 保持原顺序
plot$parameter
levels(plot$parameter) <- gsub('"', '', levels(plot$parameter))  # 去掉可能的引号字符
parameter_means <- aggregate(x ~ parameter, data = plot, FUN = mean)

# 按照均值排序
parameter_order <- parameter_means[order(parameter_means$x), "parameter"]

plot$parameter <- factor(plot$parameter, levels = parameter_order)


p <- ggplot(plot, aes(x = x, y = parameter, height = scaled_density, fill = ifelse(x >= 0, "Positive", "Negative"))) +
  geom_density_ridges(stat = "identity") +
  theme_ridges() + # 设置主题
  scale_fill_manual(values = c("Positive" = "darkred", "Negative" = "coral")) + # 设置颜色
  
  theme_publication()+
  theme(legend.position = "none",axis.title.x = element_blank(), axis.title.y = element_blank(),
    
    axis.text.x = element_text(size = 25),  # 增大字体
    axis.text.y = element_text(size = 25),  # 增大字体
  )+
  scale_y_discrete(labels = function(x) gsub("`", "", x)) 
p
getwd()
ggsave("bayes_plot.png", p, width = 12, height = 8, dpi = 600)





# ggsave("high_res_plot.png", plot = plot, dpi = 300, width = 10, height = 8)

all_params <- colnames(posterior)
params_to_keep <- setdiff(all_params, exclude_params)

posterior_filtered <- posterior[, params_to_keep]

coef_summary <- apply(posterior_filtered, 2, function(x) {
  c(
    Estimate = round(median(x), 2),
    Lower = round(quantile(x, 0.025), 2),
    Upper = round(quantile(x, 0.975), 2)
  )
})

coef_summary_df <- as.data.frame(t(coef_summary))

colnames(coef_summary_df) <- c("Estimate", "Lower", "Upper")



coef_summary_df$Est.error <- coef_summary_df$Estimate - coef_summary_df$Lower

coef_summary_df$`95%CI(Credible intervals)` <- paste(coef_summary_df$Lower, coef_summary_df$Upper, sep = "-")
rownames(coef_summary_df) <- sapply(rownames(coef_summary_df), function(x) gsub("_", " ", x))

coef_summary_df <- subset(coef_summary_df, select = -c(Lower, Upper))

coef_summary_df$Predictor <- rownames(coef_summary_df)
coef_summary_df$Response <- "Plant Disease"

print(coef_summary_df)
coef_summary_df <- coef_summary_df[, c("Response", "Predictor", "Estimate", "95%CI(Credible intervals)")]
coef_summary_df
library(openxlsx)


print(coef_summary_df)
custom_colnames <- function(colnames) {
  # 将下划线替换为空格
  # 将 95%CI 部分替换为两行
  colnames <- gsub("95%CI\\(Credible intervals\\)", "95\\% CI\\par(Credible intervals)", colnames)
  return(colnames)
}

library(stringr)
coef_summary_df$Predictor <- str_remove_all(coef_summary_df$Predictor, "`")
coef_summary_df <- coef_summary_df[order(coef_summary_df$Estimate, decreasing = FALSE), ]
write.xlsx(coef_summary_df, "coef_summary.xlsx", rowNames = FALSE)
getwd()
latex_table <- xtable(coef_summary_df,
                      align = c("l", "l", "l", "r", "r", "l"),
                      caption = "Summary of Regression Coefficients")
print(latex_table)

output_path <- "C:/Users/r/Desktop/bayes/regression_coefficients_filtered.tex"

print(latex_table,
      type = "latex",
      include.rownames = FALSE,
      sanitize.colnames.function = custom_colnames,
      file = output_path)

# 通知保存成功
cat("LaTeX table saved to:", output_path)
