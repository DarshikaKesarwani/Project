rm(list = ls())
#setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/3 NB SVM KNN")
install.packages("e1071")

install.packages("DMwR")
library(e1071)  # For NB and SVM
library(caret)
# install.packages("DMwR")
library(DMwR)    # For KNN

delays.df <- read.csv("CSV_FlightDelays.csv")

#### Classification Using Naive Bayes

# change numerical variables to categorical first
delays.df$DAY_WEEK <- factor(delays.df$DAY_WEEK)
delays.df$DEP_TIME <- factor(delays.df$DEP_TIME)
# create hourly bins departure time 
delays.df$CRS_DEP_TIME <- factor(round(delays.df$CRS_DEP_TIME/100))

head(delays.df,5)

# Create training and validation sets.
selected.var <- c("Flight.Status", "CRS_DEP_TIME", "CARRIER","DEST","DISTANCE","ORIGIN",
                  "Weather","DAY_WEEK", "DEP_TIME")
train.index <- sample(c(1:dim(delays.df)[1]), dim(delays.df)[1]*0.6)  
train.df <- delays.df[train.index, selected.var]
valid.df <- delays.df[-train.index, selected.var]

# run naive bayes
delays.nb <- naiveBayes(Flight.Status ~ ., data = train.df)
delays.nb


## predict probabilities: Training
pred.prob <- predict(delays.nb, newdata = train.df, type = "raw")
## predict class membership
pred.class <- predict(delays.nb, newdata = train.df)
confusionMatrix(pred.class, train.df$Flight.Status)

## predict probabilities: Validation
pred.prob <- predict(delays.nb, newdata = valid.df, type = "raw")
## predict class membership
pred.class <- predict(delays.nb, newdata = valid.df)
confusionMatrix(pred.class, valid.df$Flight.Status)



#### Classification using SVM
svm.model <- svm(Flight.Status ~ ., data = train.df, cost = 100, gamma = 1)

## predict class membership: Training
svm_pred_train <- predict(svm.model, newdata = train.df)
confusionMatrix(svm_pred_train, train.df$Flight.Status)

## predict class membership: Validation
svm_pred_valid <- predict(svm.model, newdata = valid.df)
confusionMatrix(svm_pred_valid, valid.df$Flight.Status)

# Train a best linear-kernel support vector classifier
linear.tune <- tune.svm(Flight.Status ~ ., data = train.df,
                        kernel = "linear",
                        cost = c(0.001, 0.01, 0.1, 1, 5, 10))
summary(linear.tune)

best.linear <- linear.tune$best.model
linear.pred <- predict(best.linear, newdata = valid.df)
confusionMatrix(linear.pred, valid.df$Flight.Status)


# Train a best polynomial-kernel support vector machine
set.seed(123) 
poly.tune <- tune.svm(Flight.Status ~ ., data = train.df,
                      kernel = "polynomial",
                      degree = c(3, 4, 5),
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4)) 
summary(poly.tune)

best.poly <- poly.tune$best.model
poly.pred <- predict(best.poly, newdata = valid.df)
confusionMatrix(poly.pred, valid.df$Flight.Status)



#### Classification using KNN
## A 3-nearest neighbours model with no normalization
Loan.df <- read.csv("CSV_UniversalBank.csv")
Loan.df$Personal.Loan <- factor(Loan.df$Personal.Loan)

# Create training and validation sets.
Loan.var <- c("Personal.Loan", "Experience", "Income","Family","Education","Mortgage",
                  "Securities.Account","CD.Account","Online","CreditCard")
train.index <- sample(c(1:dim(Loan.df)[1]), dim(Loan.df)[1]*0.6)  
train.loan <- Loan.df[train.index, Loan.var]
valid.loan <- Loan.df[-train.index, Loan.var]
nn3

# compute knn for different k on validation.
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))
for(i in 1:14) {
  kNN.pred <- kNN(Personal.Loan ~ .,train.loan,valid.loan,norm=TRUE,k=i)
  accuracy.df[i, 2] <- confusionMatrix(kNN.pred, valid.loan$Personal.Loan)$overall[1] 
}
accuracy.df

attach(accuracy.df)
accuracy.df <- accuracy.df[order(accuracy),] 
detach(accuracy.df)

# Find optimal K
set.seed(502)
grid1 <- expand.grid(.k = seq(2, 20, by = 1))
control <- trainControl(method = "cv")
knn.train <- train(Personal.Loan ~ ., data = train.loan,
                   method = "knn",
                   trControl = control,
                   tuneGrid = grid1)
knn.train


# Different distance weighting
#install.packages("kknn")
library(kknn)
set.seed(123)
kknn.train <- train.kknn(Personal.Loan ~ ., data = train.loan, kmax = 25, 
                           distance = 2, 
                           kernel = c("rectangular", "triangular", "epanechnikov"))
plot(kknn.train)

kknn.train

kknn.pred <- predict(kknn.train, newdata = valid.loan)
confusionMatrix(kknn.pred, valid.loan$Personal.Loan)


#### SVM with Plots
svm.model <- svm(Personal.Loan ~ ., data = train.loan)

## predict class membership: Training
svm_pred_train <- predict(svm.model, newdata = train.loan)
confusionMatrix(svm_pred_train, train.loan$Personal.Loan)

## predict class membership: Validation
svm_pred_valid <- predict(svm.model, newdata = valid.loan)
confusionMatrix(svm_pred_valid, valid.loan$Personal.Loan)

plot(svm.model, data=train.loan, Income ~ Mortgage)
plot(svm.model, data=train.loan, Income ~ Experience) 
plot(svm.model, data=train.loan, Income ~ Education)

# Tune the best model
svm_tuned = tune.svm(Personal.Loan ~ ., data = train.loan, gamma = 10^(-6:-1),cost = 10^(1:2))
summary(svm_tuned)

