rm(list = ls())
accident.df <- read.csv("Accidents.csv")
View(accident.df)
library(e1071)  # For NB and SVM

install.packages("caret")
library(caret)
unique(accident.df$MAX_SEV_IR)
accident.df$INJURY <- 'no'
accident.df$INJURY <- ifelse(accident.df$MAX_SEV_IR>= 1 , 'yes', 'no')
head(accident.df,5)

#Naive rule to find out the proportion of yes and no
table(accident.df$INJURY)



selected.var <- c("ALIGN_I", "WRK_ZONE", "INJURY")


#to convert the categorical column into factor so that it is treated as numerically internally
accident.df$INJURY <- factor(accident.df$INJURY)


# Create training and validation sets.
train.index <- sample(c(1:dim(accident.df)[1]), dim(accident.df)[1]*0.6)  
train.df <- accident.df[train.index, selected.var]
valid.df <- accident.df[-train.index, selected.var]
install.packages("e1071")
library(e1071)  
# run naive bayes
model.nb <- naiveBayes(INJURY ~ ., data = train.df)
model.nb



## predict probabilities: Training
pred.prob <- predict(model.nb, newdata = train.df, type = "raw")
## predict class membership
pred.prob
pred.class <- predict(model.nb, newdata = train.df)
confusionMatrix(pred.class, train.df$INJURY)

## predict probabilities: Validation
pred.prob <- predict(model.nb, newdata = valid.df, type = "raw") #it gives predition probabilit without the cutoff outcome
pred.prob
## predict class membership
pred.class <- predict(model.nb, newdata = valid.df)
confusionMatrix(pred.class, valid.df$INJURY)


table(accident.df$SPD_LIM)