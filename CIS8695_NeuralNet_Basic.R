rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/5 NN AI")
install.packages ("neuralnet")
library(neuralnet)
library(nnet)
library(caret)

accidents.df <- read.csv("CSV_Accidents_new.csv")
# selected variables
vars <- c("ALCHL_I", "PROFIL_I_R", "VEH_INVL")

# partition the data
set.seed(2)
training<-sample(row.names(accidents.df), dim(accidents.df)[1]*0.6)
validation<-setdiff(row.names(accidents.df), training)

# when y has multiple classes - need to dummify
trainData <- cbind(accidents.df[training,c(vars)], 
                   class.ind(accidents.df[training,]$SUR_COND),
                   class.ind(accidents.df[training,]$MAX_SEV_IR))
names(trainData) <- c(vars, 
                   paste("SUR_COND_", c(1, 2, 3, 4, 9), sep=""), paste("MAX_SEV_IR_", c(0, 1), sep=""))

validData <- cbind(accidents.df[validation,c(vars)], 
                   class.ind(accidents.df[validation,]$SUR_COND),
                   class.ind(accidents.df[validation,]$MAX_SEV_IR))
names(validData) <- c(vars, paste("SUR_COND_", c(1, 2, 3, 4, 9), sep=""), paste("MAX_SEV_IR_", c(0, 1), sep=""))

# run nn with 2 hidden nodes 
# use hidden= with a vector of integers specifying number of hidden nodes in each layer
nn <- neuralnet(MAX_SEV_IR_0 + MAX_SEV_IR_1 ~ 
                  ALCHL_I + PROFIL_I_R + VEH_INVL + SUR_COND_1 + SUR_COND_2 
                + SUR_COND_3 + SUR_COND_4, data = trainData, hidden = 2)

training.prediction <- compute(nn, trainData[,-c(8:11)])
training.class <- apply(training.prediction$net.result,1,which.max)-1
confusionMatrix(as.factor(training.class), as.factor(accidents.df[training,]$MAX_SEV_IR))

validation.prediction <- compute(nn, validData[,-c(8:11)])
validation.class <-apply(validation.prediction$net.result,1,which.max)-1
confusionMatrix(as.factor(validation.class), as.factor(accidents.df[validation,]$MAX_SEV_IR))


install.packages("NeuralNetTools")
library(NeuralNetTools)
# Plot neural net
par(mfcol=c(1,1))
plotnet(nn)
# get the neural weights
neuralweights(nn)
# Plot the importance
olden(nn)



