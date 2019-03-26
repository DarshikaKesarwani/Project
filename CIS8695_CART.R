rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/2 Trees")

library(rpart)
# install.packages("rpart.plot")    # Installation is required for the first of use
library(rpart.plot)

bank.df <- read.csv("CSV_UniversalBank.csv")
bank.df <- bank.df[ , -c(1, 5)]  # Drop ID and zip code columns.

# partition
set.seed(1)  
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6)  
train.df <- bank.df[train.index, ]
valid.df <- bank.df[-train.index, ]

# classification tree
default.ct <- rpart(Personal.Loan ~ ., data = train.df, method = "class")
summary(default.ct)
# define rpart.control() in rpart() to determine the depth of the tree.
default.ct <- rpart(Personal.Loan ~ ., data = train.df, 
                    control = rpart.control(maxdepth = 5), method = "class")
summary(default.ct)


# plot using prp()
prp(default.ct, type = 5, extra = 101,  clip.right.lab = FALSE, 
    box.palette = "GnYlRd", leaf.round = 5, 
    branch = .3, varlen = -10, space=0)  


## plot a fancy tree
# install.packages("rattle")  # Installation is required for the first of use
library(rattle)
fancyRpartPlot(default.ct, type=5, space=0, palettes="YlGn")


# An alternative approach using plotmo
# install.packages("plotmo")  # Installation is required for the first of use
# install.packages("TeachingDemos")  # plotmo may require the package of TeachingDemos
library(plotmo)
plotmo(default.ct)


# Look at decision rules
rpart.rules(default.ct, extra = 4, cover = TRUE)


#### Table 9.3
# classify records in the training data.
default.ct.pred.train <- predict(default.ct,train.df,type = "class")
# generate confusion matrix for training data
# install.packages("caret") # Package installation is required for the first time, it takes some time!
library(caret)
confusionMatrix(default.ct.pred.train, as.factor(train.df$Personal.Loan))
### repeat the code for the validation set, and deeper trees


# Grow a full tree, 
deeper.ct <- rpart(Personal.Loan ~ ., data = train.df, method = "class", 
                   cp = 0, minsplit = 1)
# count number of leaves
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
# plot tree
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  


# Use Complexity Parameter (CP) to determine optimal stopping of growth, 
# argument xval refers to the number of folds to use in rpart's built-in cross-validation procedure
# argument cp sets the smallest value for the complexity parameter.
cv.ct <- rpart(Personal.Loan ~ ., data = train.df, method = "class", 
               cp = 0.00001, minsplit = 5, xval = 5)
# use printcp() to print the table. 
printcp(cv.ct)



# prune tree by lower cp, 
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
prp(pruned.ct, type = 5, extra = 101, cex=0.6, box.palette = "GnYlRd", 
    branch = .3, varlen = -10, space=0, yspace=0) 



#### Find optimal CP, and build tree accordingly
set.seed(1)
cv.ct <- rpart(Personal.Loan ~ ., data = train.df, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  # minsplit is the minimum number of observations in a node for a split to be attempted. xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct)  # Print out the cp table of cross-validation errors. The R-squared for a regression tree is 1 minus rel error. xerror (or relative cross-validation error where "x" stands for "cross") is a scaled version of overall average of the 5 out-of-sample errors across the 5 folds.
pruned.ct <- prune(cv.ct, cp = 0.0154639)
prp(pruned.ct, type = 5, extra = 101, leaf.round = 5, tweak=1.2, 
    branch = .3, varlen = -10, space=0, yspace=0, box.palette = "GnYlRd") 
pruned.ct <- prune(cv.ct, cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
prp(pruned.ct, type = 5, extra = 101, leaf.round = 5, tweak=1.2, 
    branch = .3, varlen = -10, space=0, yspace=0, box.palette = "GnYlRd") 



## random forest
# install.packages("randomForest")
library(randomForest)
rf <- randomForest(as.factor(Personal.Loan) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)  
summary(rf)
head(rf$votes,10)


## Plot forest by prediction errors
plot(rf)
legend("top", colnames(rf$err.rate),cex=0.8,fill=1:3)


## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred <- predict(rf, valid.df)
# library(caret)
confusionMatrix(rf.pred, as.factor(valid.df$Personal.Loan))




# Boosting tree approach 
# install.packages("adabag")
library(adabag)

train.df$Personal.Loan <- as.factor(train.df$Personal.Loan)

set.seed(1)
boost <- boosting(Personal.Loan ~ ., data = train.df)
boost$trees[3]
pred <- predict(boost, valid.df)
confusionMatrix(as.factor(pred$class), as.factor(valid.df$Personal.Loan))


### Boosting Using GBM package
# install.packages("gbm")
library(gbm)

set.seed(2)
Loan.gbm <- gbm(formula = Personal.Loan ~ .,distribution = "bernoulli",
                data = train.df,n.trees = 1000,interaction.depth = 7,
                shrinkage = 0.01, cv.folds=3)
summary(Loan.gbm)  

Loan.iter <- gbm.perf(Loan.gbm,method="cv") 

# Prediction
Loan.predict <- predict(Loan.gbm, valid.df, n.trees = Loan.iter,type = "response")
Loan.predict.class = ifelse(Loan.predict > 0.5, 1, 0)

confusionMatrix(as.factor(Loan.predict.class), as.factor(valid.df$Personal.Loan))



