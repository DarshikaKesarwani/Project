rm(list = ls())
setwd("/Users/darsh/Documents/R/")

# Load the dataset
mortgage.df <- read.csv("loan.csv")
mortage.df <- mortgage.df[1:100000,]

# Structure of dataset
str(mortgage.df)

# Structure of dataset
summary(mortgage.df)

# Remove id, member_id columns as they are not relevant and policy_code which is all 1's
mortgage.df <- mortgage.df[, -c(1,2,52)]

# Find null values in each column
sapply(mortgage.df, function(x) sum(is.na(x)))

# Remove columns which contain more than 50% missing values
mortgage.df <- mortgage.df[, -which(colMeans(is.na(mortgage.df)) > 0.5)]

# Find loan_status distribution
table(mortgage.df$loan_status)

# Convert to binary, combine 'Default' and 'Charged Off' loan_status
mortgage.df$loan_status = factor(mortgage.df$loan_status,levels=c('Fully Paid','Default','Charged Off'),labels=c("0","1","1"))

# Find loan_status distribution
table(mortgage.df$loan_status)

# Remove all null values
mortgage.df <- na.omit(mortgage.df)

# Find which variables are categorical and numeric
split(names(mortgage.df),sapply(mortgage.df, function(x) paste(class(x), collapse=" ")))

# Correlation plot
library(corrplot)
numeric.var <- sapply(mortgage.df, is.numeric)
corr.matrix <- cor(mortgage.df[,numeric.var])
corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="circle"
         ,type="upper",order="hclust",sig.level = 0.05,insig = "blank")

# Graphs and Plots
library("ggplot2")
ggplot(mortgage.df,aes(x=mortgage.df$term)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$grade)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$sub_grade)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$emp_length)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$home_ownership)) + geom_bar()
table(mortgage.df$home_ownership)

ggplot(mortgage.df,aes(x=mortgage.df$verification_status)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$loan_status)) + geom_bar()

ggplot(mortgage.df,aes(x=mortgage.df$pymnt_plan)) + geom_bar()
table(mortgage.df$pymnt_plan)
# All values are 'n', no variance so can remove this field

ggplot(mortgage.df,aes(x=mortgage.df$purpose)) + geom_bar()
table(mortgage.df$purpose)

# single value, remove both 
ggplot(mortgage.df,aes(x=mortgage.df$application_type)) + geom_bar()
table(mortgage.df$application_type)
# No variance so can remove this field

# Combining values which are not frequent into a single factor
mortgage.df$home_ownership <- factor(mortgage.df$home_ownership,levels=c('MORTGAGE','OWN','RENT','NONE','OTHER','ANY')
                                     ,labels=c("M","O","R","OT","OT","OT"))
mortgage.df$purpose <- factor(mortgage.df$purpose,levels=c('debt_consolidation','credit_card','home_improvement','other'
                                                           ,'major_purchase','small_business', 'car', 'house', 'medical'
                                                           , 'vacation', 'moving', 'wedding', 'renewable_energy', 'educational')
                              ,labels=c("D","C","H","OT","M","S", "CA", "HO", "ME", "V", "MO", "CM", "CM", "CM"))

# Select variables
selected.var <- c('loan_amnt','term','int_rate','grade','emp_length','home_ownership',
                  'annual_inc','verification_status','purpose','dti','inq_last_6mths','loan_status')

# Partition dataset
train.index <- sample(c(1:dim(mortgage.df)[1]), dim(mortgage.df)[1]*0.7)  
train.df <- mortgage.df[train.index,selected.var ]
valid.df <- mortgage.df[-train.index,selected.var]

# Logistic Regression
logit.reg <- glm(loan_status ~ ., data = train.df, family = "binomial") 
options(scipen=999) # remove scientific notation
summary(logit.reg)

# Prediction on Validation data
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")
library(caret)
confusionMatrix(as.factor(ifelse(logit.reg.pred > 0.5, 1, 0))
                , as.factor(valid.df$loan_status), positive = "1")

# Plot ROC
library(pROC)
roc.logitreg <- roc(valid.df$loan_status, logit.reg.pred)
plot(roc.logitreg, col="red", main = "ROC Curve: Logistic Regression")

# AUC
auc(roc.logitreg)

# Rose Sampling to handle bias
library(ROSE)
rose.df <- ROSE(loan_status ~ ., data=train.df, seed=123)$data

# Logistic Regression with ROSE sampling
logit.reg.samp <- glm(loan_status ~ ., data = rose.df, family = "binomial") 
options(scipen=999) # remove scientific notation
summary(logit.reg.samp)

# Prediction on Validation data
logit.reg.samp.pred <- predict(logit.reg.samp, valid.df, type = "response")
confusionMatrix(as.factor(ifelse(logit.reg.samp.pred > 0.5, 1, 0))
                , as.factor(valid.df$loan_status), positive = "1")

# Plot ROC
library(pROC)
roc.logitreg.samp <- roc(valid.df$loan_status, logit.reg.samp.pred)
plot(roc.logitreg.samp, col="blue", main = "ROC Curve: Logistic Regression Sampled")

# AUC
auc(roc.logitreg.samp)

# Random Forest Model
library(randomForest)
rf <- randomForest(as.factor(loan_status) ~ ., data =train.df, ntree = 100, 
                   mtry = 4, nodesize = 5, importance = TRUE)  
summary(rf)
head(rf$votes,10)

# Plot forest by prediction errors
plot(rf)
legend("top", colnames(rf$err.rate),cex=0.8,fill=1:3)

# Variable Importance Plot
varImpPlot(rf, type = 1)

# Confusion Matrix
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, as.factor(valid.df$loan_status))

# Random Forest Model with ROSE Sampling
rf.samp <- randomForest(as.factor(loan_status) ~ ., data =rose.df, ntree = 100, 
                        mtry = 4, nodesize = 5, importance = TRUE)  
summary(rf.samp)
head(rf.samp$votes,10)

# Plot forest by prediction errors
plot(rf.samp)
legend("top", colnames(rf.samp$err.rate),cex=0.8,fill=1:3)

# Variable Importance Plot
varImpPlot(rf.samp, type = 1)

# Confusion Matrix
rf.samp.pred <- predict(rf.samp, valid.df)
confusionMatrix(rf.samp.pred, as.factor(valid.df$loan_status))

#KNN

library(e1071)  
library(caret)
#install.packages("DMwR")
library(DMwR)    # For KNN

setwd("/Users/darsh/Documents/R/") # Set Working directory
new.df <- read.csv("loan.csv") #Load csv File
new.df <- new.df[1:100000,] # We will take only top 100,000 rows

# Remove columns which contain more than 50% missing values
new.df <- new.df[, -which(colMeans(is.na(new.df)) > 0.5)]

# Convert to binary, combine 'Default' and 'Charged Off' loan_status
new.df$loan_status = factor(new.df$loan_status,levels=c('Fully Paid','Default','Charged Off'),
                            labels=c("0","1","1"))

# Remove all null values
new.df <- na.omit(new.df)

new.df$home_ownership <- factor(new.df$home_ownership,levels=c('MORTGAGE','OWN','RENT','NONE','OTHER','ANY')
                                ,labels=c("M","O","R","OT","OT","OT"))
new.df$purpose <- factor(new.df$purpose,levels=c('debt_consolidation','credit_card','home_improvement','other'
                                                 ,'major_purchase','small_business', 'car', 'house', 'medical'
                                                 , 'vacation', 'moving', 'wedding', 'renewable_energy', 'educational')
                         ,labels=c("D","C","H","OT","M","S", "CA", "HO", "ME", "V", "MO", "CM", "CM", "CM"))
new.df <- new.df[,c("home_ownership", "grade", "term","purpose","verification_status", "loan_amnt", 
                    "int_rate", "annual_inc","dti","inq_last_6mths", "loan_status")]

# Find which variables are categorical and numeric
split(names(new.df),sapply(new.df, function(x) paste(class(x), collapse=" "))) 

#install.packages("dummies")
library(dummies)
# Convert Categorical fields to dummy variable
new.df <- dummy_cols(new.df, select_columns = "home_ownership", remove_first_dummy = TRUE)
new.df <- dummy_cols(new.df, select_columns = "grade", remove_first_dummy = TRUE)
new.df <- dummy_cols(new.df, select_columns = "term", remove_first_dummy = TRUE)
new.df <- dummy_cols(new.df, select_columns = "purpose", remove_first_dummy = TRUE)
new.df <- dummy_cols(new.df, select_columns = "verification_status", remove_first_dummy = TRUE)
new.df <- dummy_cols(new.df, select_columns = "loan_status", remove_first_dummy = TRUE)
# remove Factor values as they are already converted to dummy
new.df <- new.df[,-c(1,2,3,4,5)]
# Find which variables are categorical and numeric
split(names(new.df),sapply(new.df, function(x) paste(class(x), collapse=" "))) 

#Seperate numeric and dummy variables
numeric.df <- new.df[,c(1,2,3,4,5)]
categoric.df <- new.df[,-c(1,2,3,4,5)]
# Normalize numeric variables
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }
numeric.df <- as.data.frame(lapply(numeric.df[1:5], normalize))
# Combine again numeric and dummy variables
new.df <- cbind(numeric.df, categoric.df)

# Partition dataset
train.knn.index <- sample(c(1:dim(new.df)[1]), dim(new.df)[1]*0.7)  
train.knn.df <- new.df[train.knn.index,]
test.knn.df <- new.df[-train.knn.index,]


str(train.knn.df)
split(names(train.knn.df),sapply(train.knn.df, function(x) paste(class(x), collapse=" ")))

# compute knn for different k on validation.
accuracy.df <- data.frame(k = seq(1, 10, 1), accuracy = rep(0, 10))
for(i in 1:10) {
  kNN.pred <- kNN(loan_status ~ .,train.knn.df,test.knn.df,norm=TRUE,k=i)
  accuracy.df[i, 2] <- confusionMatrix(kNN.pred, test.knn.df$loan_status)$overall[1] 
}
accuracy.df

attach(accuracy.df)
accuracy.df <- accuracy.df[order(accuracy),] 
detach(accuracy.df)

# Find optimal K
set.seed(502)
grid1 <- expand.grid(.k = seq(2, 20, by = 1))
control <- trainControl(method = "cv")
knn.train <- train(loan_status ~ ., data = train.knn.df,
                   method = "knn",
                   trControl = control,
                   tuneGrid = grid1)
knn.train

# Different distance weighting
#install.packages("kknn")
library(kknn)
set.seed(123)
kknn.train <- train.kknn(loan_status ~ ., data = train.knn.df, kmax = 50, 
                         distance = 2, 
                         kernel = c("rectangular", "triangular", "epanechnikov"))
plot(kknn.train)
kknn.pred <- predict(kknn.train, newdata = test.knn.df)
confusionMatrix(kknn.pred, test.knn.df$loan_status)

can somebody run this code??
 
selected.var <- c('loan_amnt','term','int_rate','annual_inc','dti','loan_status')

# Partition dataset
set.seed(12)
reduce.index <- sample(c(1:dim(mortgage.df)[1]), dim(mortgage.df)[1]*0.3)
mortgage.redu <- mortgage.df[reduce.index, selected.var]

table(mortgage.redu$loan_status)

mortgage.redu$loan_amnt <- (mortgage.redu$loan_amnt - min(mortgage.redu$loan_amnt))/ (max(mortgage.redu$loan_amnt)-min(mortgage.redu$loan_amnt))
mortgage.redu$int_rate <- (mortgage.redu$int_rate - min(mortgage.redu$int_rate))/ (max(mortgage.redu$int_rate)-min(mortgage.redu$int_rate))
mortgage.redu$annual_inc <- (mortgage.redu$annual_inc - min(mortgage.redu$annual_inc))/ (max(mortgage.redu$annual_inc)-min(mortgage.redu$annual_inc))
mortgage.redu$dti <- (mortgage.redu$dti - min(mortgage.redu$dti))/ (max(mortgage.redu$dti)-min(mortgage.redu$dti))

train.index <- sample(c(1:dim(mortgage.redu)[1]), dim(mortgage.redu)[1]*0.7)
train.df <- mortgage.redu[train.index,selected.var ]
valid.df <- mortgage.redu[-train.index,selected.var]

library(neuralnet)
nn <- neuralnet(loan_status~loan_amnt+int_rate+annual_inc+dti, data=train.knn.df, hidden = 4, stepmax=1e6)
plot(nn)

preds <- compute(nn, test.knn.df[,c('loan_amnt','int_rate','annual_inc','dti')])
preds.class <- apply(preds$net.result,1, which.max)-1
library(caret)
confusionMatrix(ifelse(preds.class=='1', 'rejected', 'approve'), train.df$loan_status)


