library(dplyr)
library(tidyverse)
library(seminr)
library(DiagrammeR)
library(glue)
plot_object <- 
  'digraph G {
    // general graph settings
    graph [
      charset = "UTF-8",
      layout = dot,
      label = "",
      fontsize = 27,
      fontname = "helvetica",
      rankdir = LR,
      labelloc = t
    ]

    // 结构模型
    subgraph sm {
      rankdir = LR;
      node [
        shape = ellipse,
        color = black,
        fillcolor = lightcyan,
        style = filled,
        fontsize = 27,
        height = 1.6,
        width = 2.5,
        fontname = "helvetica",
        fixedsize = true
      ]
      Climate
      Geography
      Soil [label="Soil\nr²=0.164"]
      PathogenLoad [label="Pathogen\nLoad\nr²=0.005"]

      // 结构模型路径（根据系数值设置线条粗细）
      edge [
        color = black,
        fontsize = 27,
        fontname = "helvetica",
        dir = both,
        arrowhead = normal,
        arrowtail = none,
        penwidth = 1  // 默认粗细
      ]
      Climate -> Soil [label="β=0.311", penwidth=2] 
      Geography -> Soil [label="β=0.804", penwidth=4]  // 系数>0.8加粗
      Climate -> PathogenLoad [label="β=-0.101", penwidth=2, style=dashed]
      Geography -> PathogenLoad [label="β=-0.089", penwidth=2, style=dashed]
      Soil -> PathogenLoad [label="β=-0.111", penwidth=2, style=dashed]
    }

    // 测量模型
    subgraph mm {
      node [
        shape = box,
        color = dimgray,
        fillcolor = lightgoldenrodyellow,
        style = filled,
        fontsize = 27,
        height = 0.8,
        width = 8,
        fontname = "helvetica",
        fixedsize = true
      ]
      // 观测变量列表...
      Solar_Radiation; Precipitation_of_Wettest_Month; /* 其他变量... */ PathogenLoad

      // 测量模型路径（根据因子负荷设置线条粗细）
      edge [
        color = blue,
        fontsize = 27,
        fontname = "helvetica",
        minlen = 1,
        dir = both,
        arrowhead = none,
        arrowtail = normal,
        penwidth = 1  // 默认粗细
      ]
      Solar_Radiation -> Climate [label=0.786, penwidth=4]
      Precipitation_of_Wettest_Month -> Climate [label=0.757, penwidth=2]
      Precipitation_Seasonality -> Climate [label=-0.743, penwidth=2, style=dashed]
      Precipitation_of_Warmest_Quarter -> Climate [label=0.78, penwidth=4]
      Precipitation_of_Coldest_Quarter -> Climate [label=0.26, penwidth=2]
      Isothermality -> Climate [label=0.112, penwidth=2]
      Min_Temperature_of_Coldest_Month -> Climate [label=0.554, penwidth=2]
      Mean_Temperature_of_Wettest_Quarter -> Climate [label=-0.51, penwidth=2, style=dashed]
      Wind_Speed -> Climate [label=-0.61, penwidth=2, style=dashed]
      Area_weighted_Soil_Organic_Carbon_Content -> Soil [label=-0.787, penwidth=2, style=dashed]
      Dominant_Mapping_Unit_ID -> Soil [label=-0.134, penwidth=2, style=dashed]
      Soil_Sand -> Soil [label=0.782, penwidth=4]
      Topsoil_Sand -> Soil [label=0.769, penwidth=4]
      Latitude -> Geography [label=0.899, penwidth=4]
      Longitude -> Geography [label=0.734, penwidth=2]
      Height_Above_the_Nearest_Drainage -> Geography [label=-0.772, penwidth=2, style=dashed]
    }
}' %>% 
  grViz()

plot_object
plot(pls_model)
# 保存图形（需要DiagrammeRsvg和rsvg包）
getwd()
plot_object %>% 
  DiagrammeRsvg::export_svg() %>% 
  charToRaw() %>% 
  rsvg::rsvg_png("path_model.png", width = 1200, height = 600)

