rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/5 Dimension Reduction")

# PCA on MNIST data
# Download MNIST datasets
dataDirectory <- "./data"
if (!file.exists(paste(dataDirectory,'/train.csv',sep="")))
{
  link <- 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip'
  if (!file.exists(paste(dataDirectory,'/mnist_csv.zip',sep="")))
    download.file(link, destfile = paste(dataDirectory,'/mnist_csv.zip',sep=""))
  unzip(paste(dataDirectory,'/mnist_csv.zip',sep=""), exdir = dataDirectory)
  if (file.exists(paste(dataDirectory,'/mnist_csv.zip',sep="")))
    file.remove(paste(dataDirectory,'/mnist_csv.zip',sep=""))
}

digits <- read.csv("./data/train.csv")
dim(digits)
head(colnames(digits), 4)
tail(colnames(digits), 4)
head(digits[, 1:4])

# visualize the digits
par(mfrow = c(3, 3))
par(mar=c(2,2,2,2))
for (i in 1:9)
{
  row <- as.numeric(digits[i,2:ncol(digits)])
  mat <- matrix(row,nrow=28,byrow=TRUE)
  mat <- t(apply(mat, 2, rev))
  image(mat, main = paste("index:",i,", label =",digits[i,1]),
        axes = FALSE, col = grey(seq(0, 1, length = 256)))
}

# To detect 5 and 6
digits_nb <- digits[(digits$label==5) | (digits$label==6),]
digits_nb$y <- digits_nb$label
digits_nb$label <- NULL
table(digits_nb$y)
digits_nb$y <- ifelse(digits_nb$y==5, 0, 1)
table(digits_nb$y)

digits.X2 <- digits_nb[,apply(digits_nb[,1:(ncol(digits_nb)-1)], 2, var, na.rm=TRUE) != 0]
length(digits.X2)

df.pca <- prcomp(digits.X2[,],center = TRUE,scale. = TRUE) 
s<-summary(df.pca)
cumprop<-s$importance[3, ]
par(mfrow=c(1,1))
plot(cumprop, type = "l",main="Cumulative sum",xlab="PCA component")

num_cols <- min(which(cumprop>0.5))
cumprop[num_cols]
newdat<-data.frame(df.pca$x[,1:num_cols])
newdat$y<-digits_nb[,"y"]

newdat$y <- factor(newdat$y)

set.seed(42)
train.index <- sample(nrow(newdat), nrow(newdat)*0.6)  
train.df <- newdat[train.index, ]
valid.df <- newdat[-train.index, ]

library(e1071)
library(caret)
nb.model <- naiveBayes(y~., data = train.df)
nb.pred <- predict(nb.model, newdata = valid.df)
confusionMatrix(nb.pred, valid.df$y)

svm.model <- svm(y ~ ., data = train.df, cost = 1)
svm.pred <- predict(svm.model, newdata = valid.df)
confusionMatrix(svm.pred, valid.df$y)


# SVD 

# install.packages("bmp")
library(bmp)

lenna = read.bmp("lena512.bmp")

lenna = t(lenna)[,nrow(lenna):1]
image(lenna) 

lenna.svd = svd(scale(lenna))
plot(lenna.svd$d^2/sum(lenna.svd$d^2), type="l", xlab=" Singular vector", 
     ylab = "Variance explained")

length(lenna.svd$d)

min(which(cumsum(lenna.svd$d^2/sum(lenna.svd$d^2))> 0.9))

lenna_compression = function(dim){
     u=as.matrix(lenna.svd$u[, 1:dim])
     v=as.matrix(lenna.svd$v[, 1:dim])
     d=as.matrix(diag(lenna.svd$d)[1:dim, 1:dim])
     image(u%*%d%*%t(v))
}

lenna_compression(18)

min(which(cumsum(lenna.svd$d^2/sum(lenna.svd$d^2))> 0.99))

lenna_compression(92)


