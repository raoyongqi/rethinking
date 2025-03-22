library(tidyverse)
library(ggsoiltexture)
rm(list=ls())
library(ggtern)
data(Feldspar)
Feldspar$Ab <- Feldspar$Ab+1
Feldspar$Ab
ggtern(data=Feldspar,aes(Ab,An,Or)) +
  geom_point()   + #Layer
  theme_bw()     + #For clarity
  theme_hidegrid() #Turn off both major and minor
ggsoiltexture(some_data)
devtools::install_github("ms609/Ternary", args = "--recursive")
