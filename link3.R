
rm(list = ls())

library(vegan)
library(linkET)
library(ggplot2)


file_path = "data/selection.csv"

data <- read.csv(file_path)

library(dplyr)

data <- data %>%
  rename(`Pathogen Load` = pathogen.load)

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

xxxx <- data[, other_columns]

yyyy <- data$`Pathogen Load`

colnames(xxxx)


Climate <- c("srad", "bio_13", "bio_15", "bio_18", "bio_19", 
             "bio_3", "bio_6", "bio_8", "wind")

Soil <- c("awt soc", "dom mu", "s_sand", "t_sand")

Geo <- c("lat", "lon", "hand")

# 假设 df 是你原来的数据框
# 根据列列表重新排序
xxxx <- xxxx %>%
  select(all_of(c(Climate, Soil, Geo)))
rownames(xxxx) <- gsub("_| ", "\n", rownames(xxxx))
colnames(xxxx) <- gsub("_| ", "\n", colnames(xxxx))
# 使用 mantel_test 进行 Mantel 检验
mantel <- mantel_test(yyyy, xxxx,
                      spec_select = list(`Pathogen Load` = 1:1)
)%>% 
  mutate(rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf),
                  labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
         pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf),
                  labels = c("< 0.01", "0.01 - 0.05", ">= 0.05")))
mantel_subset <- mantel[, 1:4]

library(openxlsx)

# 提取前 4 列

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
cor_matrix <- correlate(xxxx)

# # 转换为数据框，并修改行名
# 
qcorrplot(cor_matrix, type = "lower", diag = FALSE)+
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), 
              data = mantel, 
              curvature = nice_curvature()) +
  scale_colour_manual(values = color_pal(3))+   scale_fill_gradientn(colours = RColorBrewer::brewer.pal(11, "RdBu"))+
  scale_size_manual(values = c(5, 5, 5))+theme_publication() +
  guides(
    size = guide_legend(
      title = "Mantel's r",
      override.aes = list(colour = "grey35", size = 5), 
      order = 2,
      size = NULL,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12)   # 增加标签字体大小
    ),
    colour = guide_legend(
      title = "Mantel's p", 
      override.aes = list(shape = 21), 
      order = 1,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12),
      theme = theme(
        keyheight = 10,
        keywidth = 10,
        default.unit = "inch"
      )           # 设置颜色图例的高度
      ),
    fill = guide_colorbar(
      title = "Pearson's r", 
      order = 3,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12)   # 增加标签字体大小
    )
  )+
  theme(
    axis.title = element_blank(),  # 隐藏 x 和 y 轴标题
    
    axis.text.x = element_text(size = 15),  # 增大字体
    axis.text.y = element_text(size = 15),  # 增大字体
    strip.text = element_text(size = 15),  # 增大字体
    panel.border = element_rect(size = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(size = .5, color = "#f0f0f0"),
    plot.margin = grid::unit(c(0, 0.1, 0.2, 0), units = "cm")
  )


# 修改列名
p <- qcorrplot(cor_matrix, type = "lower", diag = FALSE)+
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), 
              data = mantel, 
              curvature = nice_curvature()) +
  scale_fill_gradientn(colours = RColorBrewer::brewer.pal(11, "RdBu")) +
  scale_size_manual(values = c(0.5, 1, 2)) +
  scale_colour_manual(values = color_pal(3))+theme_publication() +
  guides(
    size = guide_legend(
      title = "Mantel's r",
      override.aes = list(colour = "grey35"), 
      order = 2,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12)   # 增加标签字体大小
    ),
    colour = guide_legend(
      title = "Mantel's p", 
      override.aes = list(size = 3), 
      order = 1,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12)   # 增加标签字体大小
    ),
    fill = guide_colorbar(
      title = "Pearson's r", 
      order = 3,
      title.theme = element_text(size = 14),  # 增加标题字体大小
      label.theme = element_text(size = 12)   # 增加标签字体大小
    )
  )+
  theme(
    axis.title = element_blank(),  # 隐藏 x 和 y 轴标题
    
    axis.text.x = element_text(size = 15),  # 增大字体
    axis.text.y = element_text(size = 15),  # 增大字体
    strip.text = element_text(size = 15),  # 增大字体
    panel.border = element_rect(size = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(size = .5, color = "#f0f0f0"),
    plot.margin = grid::unit(c(0, 0.1, 0.2, 0), units = "cm")
  )


p
# 获取图形对象的 grob
gt <- ggplotGrob(p)

# 查看图形元素的布局
cor_matrix2 <- correlate(xxxx)

dimnames(cor_matrix2$r)[[1]] <- gsub("\n", " ", dimnames(cor_matrix2$r)[[1]])
dimnames(cor_matrix2$p)[[1]] <- gsub("\n", " ", dimnames(cor_matrix2$p)[[1]])

p2 <-qcorrplot(cor_matrix2,
)+
  theme(
    axis.title = element_blank(),  # 隐藏 x 和 y 轴标题
    
    axis.text.x = element_text(size = 15),  # 增大字体
    axis.text.y = element_text(size = 15),  # 增大字体
    strip.text = element_text(size = 15),  # 增大字体
    panel.border = element_rect(size = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(size = .5, color = "#f0f0f0"),
    plot.margin = grid::unit(c(0, 0.1, 0.2, 0), units = "cm")
  )

gt2 <- ggplotGrob(p2)

gt$grobs[[3]] <- gt2$grobs[[3]]
library(grid)
library(gridExtra)
png("your_plot.png", width = 12, height = 8, units = "in", res = 600)

# 绘制 grid 对象
grid.draw(gt)

# 关闭图形设备
dev.off()
traceback()
# 在 R 启动时增加 C stack 的大小
options(expressions = 500000)  # 增加最大表达式数限制
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
