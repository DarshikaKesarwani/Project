rm(list = ls())
setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/1 Regression")
# options(repos = c(CRAN = "http://cran.rstudio.com"))

bank.df <- read.csv("CSV_UniversalBank.csv")
bank.df <- bank.df[ , -c(1, 5)]  # Drop ID and zip code columns.

# treat Education as categorical (R will create dummy variables)
bank.df$Education <- factor(bank.df$Education, levels = c(1, 2, 3), 
                            labels = c("Undergrad", "Graduate", "Advanced/Professional"))

# If you want to create dummies explicitly
# install.packages("dummies")
library(dummies)
bank.df<-data.frame(Education=bank.df$Education,dummy.data.frame(bank.df, names = "Education",sep=".",dummy.classes =c("Undergrad", "Graduate", "Advanced/Professional")))

# Select variables
selected.var <- c(1:6, 10:15)

# partition data
set.seed(2)
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6)
train.df <- bank.df[train.index, selected.var]
valid.df <- bank.df[-train.index, selected.var]

# run logistic regression
# use glm() (general linear model) with family = "binomial" to fit a logistic 
# regression.
logit.reg <- glm(Personal.Loan ~ ., data = train.df, family = "binomial") 
options(scipen=999) # remove scientific notation
summary(logit.reg)
# install.packages("jtools")    # jtools helps present regression results
library(jtools)
summ(logit.reg, digits=5)

# use predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")
# use linear regression for comparison
bank.lm <- lm(Personal.Loan ~ ., data = train.df)
bank.lm.pred <-predict(bank.lm, valid.df)

# actual and predicted records
Plot.Pred<-data.frame(actual = valid.df$Personal.Loan, logit.pred = logit.reg.pred, lm.pred=bank.lm.pred)

# plot actual and predicted values
Plot.Pred<-Plot.Pred[order(Plot.Pred$logit.pred),]
Plot.Pred$x<-c(seq(1,2000))
with(Plot.Pred, plot(x, actual, col="red3"))
par(new = T)
with(Plot.Pred, plot(x, logit.pred, col="black"))
par(new = T)
with(Plot.Pred, plot(x, lm.pred, col="blue"))

# Use Confusion Matrix to evaluate performance
# install.packages("caret") # Package installation is required for the first time, it takes some time!
library(caret)
confusionMatrix(as.factor(ifelse(logit.reg.pred > 0.5, 1, 0)), as.factor(valid.df$Personal.Loan))
# Map into TP, FP, TN, FN
confusionMatrix(as.factor(ifelse(logit.reg.pred > 0.5, "Positive", "Negative")), as.factor(ifelse(valid.df$Personal.Loan==1,"Positive", "Negative")))

# Variable Selection
# Forward
null<-glm(Personal.Loan ~ 1, data = train.df, family = "binomial") 
step(null, scope=list(lower=null, upper=logit.reg), direction="forward")

# Outlier detection using Cook's Distance
cooksd <- cooks.distance(logit.reg)

plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels

# K-fold cross validation
bank.df$Loan2 <- factor(bank.df$Personal.Loan, levels = c(0,1))
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
mod_fit <- train(Loan2 ~ Age+Experience+Income+Family+CCAvg+Education+Mortgage, data=bank.df, method="glm", family="binomial",
                 trControl = ctrl)
summary(mod_fit)
mod_fit$pred

