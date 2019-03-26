rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/5 NN AI")

#==================================
#cran <- getOption("repos")
#cran["dmlc"] <- "https://s3.amazonaws.com/mxnet-r/"
#options(repos = cran)
#install.packages("mxnet")
##
#library(devtools)
#devtools::install_github('rich-iannone/DiagrammeR')
##
## install.packages("drat", repos="https://cran.rstudio.com")
## drat:::addRepo("dmlc")
## install.packages("mxnet")
## install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos = NULL)
## 
## #Please refer https://github.com/dahtah/imager
## install.packages("devtools")
## devtools::install_github("dahtah/imager",dependencies=TRUE)
## install.packages("imager")
## install.packages("DiagrammeR")

library(mxnet)

#install imager for loading images
library(imager)

#load the pre-trained model
model <- mx.model.load("Inception/Inception_BN", iteration=39)

#We also need to load in the mean image, which is used for preprocessing using mx.nd.load.

mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

#Load and plot the image: (Defualt parot image)

#im <- load.image("Images/russia-volcano.jpg")
im <- load.image("Images/elephant.jpg")
plot(im)

preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

#Now pass our image to pre-process
normed <- preproc.image(im, mean.img)
plot(normed)

prob <- predict(model, X=normed)

#We can extract the top-5 class index.

max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
max.idx

synsets <- readLines("Inception/synset.txt")

#And let us print the corresponding lines:

print(paste0("Predicted Top-classes: ", synsets[as.numeric(max.idx)]))
