rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/4 Ensemble")

library(adabag)
library(rpart) 
library(caret)
# install.packages("klaR")
library(klaR)

bank.df <- read.csv("CSV_UniversalBank.csv")
bank.df <- bank.df[ , -c(1, 5)]  # Drop ID and zip code columns.

# transform Personal.Loan into categorical variable
bank.df$Personal.Loan = factor(bank.df$Personal.Loan,levels=c(0,1),labels=c("0","1"))
# Make class levels valid R variable names 
levels(bank.df$Personal.Loan) <- make.names(levels(factor(bank.df$Personal.Loan)))


# partition the data
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6)  
train.df <- bank.df[train.index, ]
valid.df <- bank.df[-train.index, ]


# Build an Ensemble Model with Multiple Types of Models
# Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#Defining the predictors and outcome
predictors<-c("Age", "Income", "Experience", "Securities.Account", "CD.Account", "CreditCard")
outcomeName<-c("Personal.Loan")

#Training a random forest model
model_rf<-train(train.df[,predictors],train.df[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)
#Predicting using random forest model
valid.df$pred_rf<-predict(object = model_rf,valid.df[,predictors])
valid.df$pred_rf.prob<-predict(object = model_rf,valid.df[,predictors],type="prob")
#Checking the accuracy of the random forest model
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_rf)

#Training a Logistic regression model
model_lr<-train(train.df[,predictors],train.df[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)
#Predicting using logistic model
valid.df$pred_lr<-predict(object = model_lr,valid.df[,predictors])
valid.df$pred_lr.prob<-predict(object = model_lr,valid.df[,predictors],type="prob")
#Checking the accuracy of the logistic model
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_lr)


#Training a Naive Bayes model
model_nb<-train(train.df[,predictors],train.df[,outcomeName],method='nb',trControl=fitControl,tuneLength=3)
#Predicting using Naive Bayes model
valid.df$pred_nb<-predict(object = model_nb,valid.df[,predictors])
valid.df$pred_nb.prob<-predict(object = model_nb,valid.df[,predictors],type="prob")
#Checking the accuracy of the Naive Bayes model
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_nb)


library(gains)
valid.df$Personal.Loan.n = ifelse(valid.df$Personal.Loan == "X1", 1, 0)
valid.df$pred_rf.n = ifelse(valid.df$pred_rf == "X1", 1, 0)
valid.df$pred_lr.n = ifelse(valid.df$pred_lr == "X1", 1, 0)
valid.df$pred_nb.n = ifelse(valid.df$pred_nb == "X1", 1, 0)

# Life chart: the main idea
lift.df <- valid.df$pred_rf.prob$X1
lift.df <- data.frame(lift.df, valid.df[,c("pred_rf.n","Personal.Loan.n")])
colnames(lift.df) <- c("pred_rf.prob.1","pred_rf.n","Personal.Loan.n")
attach(lift.df)
lift.df <- lift.df[order(-pred_rf.prob.1),]
detach(lift.df)
lift.df$Personal.Loan.n.cum<-cumsum(lift.df$Personal.Loan.n)
lift.df[1500:1550,]
lift.df[1980:2000,]


# Gains: Random Forest
gain.rf <- gains(valid.df$Personal.Loan.n, valid.df$pred_rf.prob$X1, groups=10)
# Gains: Logistic Regression
gain.lr <- gains(valid.df$Personal.Loan.n, valid.df$pred_lr.prob$X1, groups=10)
# Gains: Naive Bayes
gain.nb <- gains(valid.df$Personal.Loan.n, valid.df$pred_nb.prob$X1, groups=10)


# Plot lift charts
plot(c(0, gain.rf$cume.pct.of.total*sum(valid.df$Personal.Loan.n)) ~ c(0, gain.rf$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l", col="red")
par(new=TRUE)
plot(c(0, gain.lr$cume.pct.of.total*sum(valid.df$Personal.Loan.n)) ~ c(0, gain.lr$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l", col="green")
par(new=TRUE)
plot(c(0, gain.nb$cume.pct.of.total*sum(valid.df$Personal.Loan.n)) ~ c(0, gain.nb$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l", col="blue")
lines(c(0,sum(valid.df$Personal.Loan.n))~c(0,dim(valid.df)[1]), col="gray", lty=2)


# compute deciles and plot decile-wise chart
par(mfrow=c(1,3))
dec.rf <- gain.rf$mean.resp/mean(valid.df$Personal.Loan.n)
barplot(dec.rf, names.arg = gain.rf$depth, ylim = c(0,9), 
        xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise: Random Forest")
dec.lr <- gain.lr$mean.resp/mean(valid.df$Personal.Loan.n)
barplot(dec.lr, names.arg = gain.lr$depth, ylim = c(0,9), 
        xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise: Logistic Regression")
dec.nb <- gain.nb$mean.resp/mean(valid.df$Personal.Loan.n)
barplot(dec.lr, names.arg = gain.nb$depth, ylim = c(0,9), 
        xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise: Naive Bayes")


# ROC
# install.packages("pROC")
library(pROC)
roc.rf <- roc(valid.df$Personal.Loan.n, valid.df$pred_rf.prob$X1)
roc.lr <- roc(valid.df$Personal.Loan.n, valid.df$pred_lr.prob$X1)
roc.nb <- roc(valid.df$Personal.Loan.n, valid.df$pred_nb.prob$X1)

plot(roc.rf,col="red")
par(new=TRUE)
plot(roc.lr,col="green")
par(new=TRUE)
plot(roc.nb,col="blue")

auc(roc.rf)
auc(roc.lr)
auc(roc.nb)


# Ensemble using Averaging
#Predicting the probabilities
valid.df$pred_rf_prob<-predict(object = model_rf,valid.df[,predictors],type='prob')
valid.df$pred_lr_prob<-predict(object = model_lr,valid.df[,predictors],type='prob')
valid.df$pred_nb_prob<-predict(object = model_nb,valid.df[,predictors],type='prob')
#Taking average of predictions
valid.df$pred_avg<-(valid.df$pred_rf_prob$X1+valid.df$pred_lr_prob$X1+valid.df$pred_nb_prob$X1)/3
#Splitting into binary classes at 0.5
valid.df$pred_avg<-as.factor(ifelse(valid.df$pred_avg>0.5,'X1','X0'))
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_avg)


# Ensemble using Majority Voting
valid.df$pred_majority<-as.factor(ifelse(valid.df$pred_rf=='X1' & valid.df$pred_nb=='X1','X1',
                          ifelse(valid.df$pred_rf=='X1' & valid.df$pred_lr=='X1','X1',
                          ifelse(valid.df$pred_nb=='X1' & valid.df$pred_lr=='X1','X1','X0'))))
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_majority)


# Ensemble using Weighted Average
# Taking weighted average of predictions
valid.df$pred_weighted_avg<-(valid.df$pred_rf_prob$X1*0.25)+(valid.df$pred_lr_prob$X1*0.5)+(valid.df$pred_nb_prob$X1*0.25)
#Splitting into binary classes at 0.5
valid.df$pred_weighted_avg<-as.factor(ifelse(valid.df$pred_weighted_avg>0.5,'X1','X0'))
confusionMatrix(valid.df$Personal.Loan,valid.df$pred_weighted_avg)






# Uplift Model
# install.packages("uplift")
library(uplift)
voter.df <- read.csv("CSV_Voter-Persuasion.csv")
# transform variable MOVED_AD to numerical
voter.df$MOVED_AD_NUM <- ifelse(voter.df$MOVED_AD == "Y", 1, 0)

set.seed(1)  
train.index <- sample(c(1:dim(voter.df)[1]), dim(voter.df)[1]*0.6)  
train.df <- voter.df[train.index, ]
valid.df <- voter.df[-train.index, ]

# use upliftRF to apply a Random Forest.
up.fit <- upliftRF(MOVED_AD_NUM ~ AGE + NH_WHITE + COMM_PT + H_F1 + REG_DAYS+ 
                     PR_PELIG + E_PELIG + POLITICALC  + trt(MESSAGE_A),
                   data = train.df, mtry = 3, ntree = 100, split_method = "KL",
                   minsplit = 200, verbose = TRUE)
pred <- predict(up.fit, newdata = valid.df)
# first colunm: p(y | treatment) 
# second colunm: p(y | control) 
head(data.frame(pred, "uplift" = pred[,1] - pred[,2]))







