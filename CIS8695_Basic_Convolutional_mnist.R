#rm(list = ls())
#setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/5 NN AI")

install.packages("visNetwork")
install.packages("mxnet")

cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet",dependencies = T)
install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos = NULL)
library(mxnet)
require(mxnet)
options(scipen=999)

# Load data, data partition
dfMnist <- read.csv("train.csv", header=TRUE)
dim(dfMnist)
yvars <- dfMnist$label
dfMnist$label <- NULL

set.seed(12)
train <- sample(nrow(dfMnist),0.9*nrow(dfMnist))
test <- setdiff(seq_len(nrow(dfMnist)),train)
train.y <- yvars[train]
test.y <- yvars[test]
train <- data.matrix(dfMnist[train,])
test <- data.matrix(dfMnist[test,])

rm(dfMnist,yvars)
tail(train[,1:6])
tail(train[,(ncol(train)-5):ncol(train)])

# Plot pictures
plotInstance <-function (row,title="")
{
  mat <- matrix(row,nrow=28,byrow=TRUE)
  mat <- t(apply(mat, 2, rev))
  image(mat, main = title,axes = FALSE, col = grey(seq(0, 1, length = 256)))
}
par(mfrow = c(3, 3))
par(mar=c(2,2,2,2))
for (i in 1:9)
{
  row <- as.numeric(train[i,2:ncol(train)])
  plotInstance(row, paste("index:",i,", label =",train[i,1]))
}


# Convert pixel values to 0-1
train <- t(train / 255.0)
test <- t(test / 255.0)
table(train.y)


# A simple NN with no convolutional layer
data <- mx.symbol.Variable("data")
fullconnect1 <- mx.symbol.FullyConnected(data, name="fullconnect1", num_hidden=256)
activation1  <- mx.symbol.Activation(fullconnect1, name="activation1", act_type="relu")
fullconnect2 <- mx.symbol.FullyConnected(activation1, name="fullconnect2", num_hidden=128)
activation2  <- mx.symbol.Activation(fullconnect2, name="activation2", act_type="relu")
fullconnect3 <- mx.symbol.FullyConnected(activation2, name="fullconnect3", num_hidden=10)
softmax      <- mx.symbol.SoftmaxOutput(fullconnect3, name="softmax")


# Model Training: A simple NN with no convolutional layer
devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train, y=train.y,
                                     ctx=devices,array.batch.size=128,
                                     num.round=10,
                                     learning.rate=0.05, momentum=0.9,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(1))

# Prediction
preds1 <- predict(model, test)
pred.label1 <- max.col(t(preds1)) - 1
res1 <- data.frame(cbind(test.y,pred.label1))
table(res1)

accuracy1 <- sum(res1$test.y == res1$pred.label1) / nrow(res1)
accuracy1


# A NN with Convolutional Layers 
data <- mx.symbol.Variable('data')
# first convolution layer
convolution1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=64)
activation1 <- mx.symbol.Activation(data=convolution1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=activation1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# second convolution layer
convolution2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=32)
activation2 <- mx.symbol.Activation(data=convolution2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=activation2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# flatten layer and then fully connected layers
flatten <- mx.symbol.Flatten(data=pool2)
fullconnect1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=512)
activation3 <- mx.symbol.Activation(data=fullconnect1, act_type="relu")
fullconnect2 <- mx.symbol.FullyConnected(data=activation3, num_hidden=10)
# final softmax layer
softmax <- mx.symbol.SoftmaxOutput(data=fullconnect2)


# Reshape the data so that it can be used in MXNet
train.array <- train
dim(train.array) <- c(28,28,1,ncol(train))
test.array <- test
dim(test.array) <- c(28,28,1,ncol(test))


# Model Training: A NN with Convolutional Layers
devices <- mx.cpu()
mx.set.seed(0)
model2 <- mx.model.FeedForward.create(softmax, X=train.array, y=train.y,
                                      ctx=devices,array.batch.size=128,
                                      num.round=10,
                                      learning.rate=0.05, momentum=0.9, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(1))

# Prediction
preds2 <- predict(model2, test.array)
pred.label2 <- max.col(t(preds2)) - 1
res2 <- data.frame(cbind(test.y,pred.label2))
table(res2)

accuracy2 <- sum(res2$test.y == res2$pred.label2) / nrow(res2)
accuracy2

# Visualize
graph.viz(model2$symbol)