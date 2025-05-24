


library(tidyverse)
library(seminr)
library(DiagrammeR)
library(glue)
source("C:/Users/r/Desktop/rethink_resample/seminrvis.R")

colnames(data) <- c(
  "Latitude",                                   # lat
  "Longitude",                                  # lon
  "Area-weighted Soil Organic Carbon Content",  # awt soc
  "Dominant Mapping Unit ID",                   # dom mu
  "Soil Sand",                                  # s_sand
  "Topsoil Sand",                               # t_sand
  "Height Above the Nearest Drainage",          # hand
  "Solar Radiation",                            # srad
  "Precipitation of Wettest Month",             # bio_13
  "Precipitation Seasonality",                  # bio_15
  "Precipitation of Warmest Quarter",           # bio_18
  "Precipitation of Coldest Quarter",           # bio_19
  "Isothermality",                              # bio_3
  "Min Temperature of Coldest Month",           # bio_6
  "Mean Temperature of Wettest Quarter",        # bio_8
  "Wind Speed",                                 # wind
  "Pathogen Load"                               # Pathogen Load
)


# the plottting function ----


plot_model <- function(model, use_outer_weights = FALSE,
                       title = "",
                       title_font = "helvetica",
                       title_size = 20,
                       theme = create_theme()
){
  
  sm_nodes <- getSMnodes(model)
  mm_nodes <- getMMnodes(model)
  sm_edges <- getSMedges(model)
  mm_edges <- getMMedges(model, use_outer_weights = use_outer_weights)
  
  
  # TODO Refactor style function to genearte list object
  #      write function to convert list objects to dot-string
  #      extract font sizes to update theme widths + heights
  model$constructs %>% strwidth(.,font = 9, units = "in") %>% max()
  
  
  # use rank=same; A; B; C; to force same level on items of construct.
  
  glue(
    "digraph G {{
      // general graph settings
      graph [
        charset = 'UTF-8',
        layout = dot,
        label = '{title}',
        fontsize = {title_size},
        fontname = {title_font},
        rankdir = LR,
        labelloc = t,
        //splines = false
      ]

      // The structural model
      subgraph sm {{
        rankdir = LR;
        node [
          {theme$construct_style}
        ]
        {sm_nodes}

        // How constructs are connected
        edge [
        {theme$inner_weight_style}
        ]
        {sm_edges}
      }

      // ---------------------
      // The measurement model
      // ---------------------
      subgraph mm {{
        node [
          {theme$item_style}
        ]
        {mm_nodes}

        // How items are connected with nodes
        edge [
        {theme$outer_weight_style}
        ]
        {mm_edges}
      }
  }"
  )
}







plot_model <- function(model, use_outer_weights = FALSE,
                       title = "",
                       title_font = "helvetica",
                       title_size = 24,
                       theme = create_theme()
                       ){

  sm_nodes <- getSMnodes(model)
  mm_nodes <- getMMnodes(model)
  sm_edges <- getSMedges(model)
  mm_edges <- getMMedges(model, use_outer_weights = use_outer_weights)


  # TODO Refactor style function to genearte list object
  #      write function to convert list objects to dot-string
  #      extract font sizes to update theme widths + heights
  model$constructs %>% strwidth(.,font = 9, units = "in") %>% max()


  # use rank=same; A; B; C; to force same level on items of construct.

  glue(
    "digraph G {{
      // general graph settings
      graph [
        charset = 'UTF-8',
        layout = dot,
        label = '{title}',
        fontsize = {title_size},
        fontname = {title_font},
        rankdir = LR,
        labelloc = t,
        //splines = false
      ]

      // The structural model
      subgraph sm {{
        rankdir = LR;
        node [
          {theme$construct_style}
        ]
        {sm_nodes}

        // How constructs are connected
        edge [
        {theme$inner_weight_style}
        ]
        {sm_edges}
      }

      // ---------------------
      // The measurement model
      // ---------------------
      subgraph mm {{
        node [
          {theme$item_style}
        ]
        {mm_nodes}

        // How items are connected with nodes
        edge [
        {theme$outer_weight_style}
        ]
        {mm_edges}
      }
  }"
  )
}

columns <- c("lat", "lon", "awt_soc", "dom_mu", "s_sand", "t_sand", 
             "hand", "srad", "bio_13", "bio_15", "bio_18", "bio_19", 
             "bio_3", "bio_6", "bio_8", "wind")


Climate <- c("srad", "bio_13", "bio_15", "bio_18", "bio_19", 
             "bio_3", "bio_6")

Soil <- c("awt_soc", "dom_mu", "s_sand", "t_sand")

Geography <- c("lat", "lon", "hand")


measurements <- constructs(
  composite("Climate", Climate),  # 定义反射性构念 "Climate"
  composite("Soil", Soil),  # 定义复合构念 "Soil"
  composite("Geography", Geography),  
  single_item("Pathogen_Load", "Pathogen_Load"))


structure <- relationships(
  paths(from = c("Climate", "Geography"), to = "Soil"),  # 气候和地理影响土壤
  
  paths(from = c("Climate", "Geography", "Soil"), to = "Pathogen Load")  # 气候、地理和土壤共同影响 Pathogen Load
)

# my_theme <- create_theme(item_style = createItemStyle(
#   fontsize = 12, fill = "lightgoldenrodyellow"),
#   
#   # we can style the construct appearance
#   construct_style = createConstructStyle(
#     fontsize = 12, fill = "lightcyan"),
#   
#   # # we can style the outer weight edges
#   # outer_weight_style = createOuterWeightStyle(color = "dimgray"),
#   # 
#   # # we can style the inner weight edges
#   # inner_weight_style = createInnerWeightStyle(color = "black", fontsize = 12)
# )
# 
pls_model %>% plot_model(
  # we can have a title
  title = "PLS-SEM Plot with interactions",

  # we can style the item appearance
  theme = my_theme
) %>%
  grViz()
  