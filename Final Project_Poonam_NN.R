rm(list = ls())
setwd("~/Desktop/MSIS/BIG DATA ANALYTICS Section 003 Spring Semester 2019 - 172019 - 613 PM")

# Load the dataset
mortgage.df <- read.csv("loan.csv")

mortgage.df <- mortgage.df[1:100000, ]

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
mortgage.df$loan_status <- factor(mortgage.df$loan_status,levels=c('Fully Paid','Default','Charged Off'),labels=c("0","1","1"))

# Find loan_status distribution
table(mortgage.df$loan_status)

mortgage.df$loan_status <- as.numeric(as.character(mortgage.df$loan_status))

# Remove all null values
mortgage.df <- na.omit(mortgage.df)

# Combining values which are not frequent into a single factor
mortgage.df$home_ownership <- factor(mortgage.df$home_ownership,levels=c('MORTGAGE','OWN','RENT','NONE','OTHER','ANY')
                                     ,labels=c("M","O","R","OT","OT","OT"))
# Select variables
selected.var <- c('loan_amnt','term','int_rate','grade','home_ownership',
                  'annual_inc','verification_status','dti','inq_last_6mths','loan_status')

mortgage.df <- mortgage.df[, selected.var]

library(dummies)
mortgage.df<-data.frame(grade=mortgage.df$grade
                        ,dummy.data.frame(mortgage.df, names = "grade",sep=".",dummy.classes =c("A", "B", "C", "D", "E", "F", "G")))
mortgage.df$grade <- NULL

mortgage.df<-data.frame(term=mortgage.df$term
                        ,dummy.data.frame(mortgage.df, names = "term",sep=".",dummy.classes =c('36 months','60 months')))
mortgage.df$term <- NULL

mortgage.df<-data.frame(home_ownership=mortgage.df$home_ownership
                        ,dummy.data.frame(mortgage.df, names = "home_ownership",sep=".",dummy.classes =c('M','O', 'R', 'OT')))
mortgage.df$home_ownership <- NULL

mortgage.df<-data.frame(verification_status=mortgage.df$verification_status
                        ,dummy.data.frame(mortgage.df, names = "verification_status",sep="."
                                          ,dummy.classes =c("Not Verified", "Source Verified", "Verified")))
mortgage.df$verification_status <- NULL

library(neuralnet)
library(nnet)
library(caret)

# Check Range
apply(mortgage.df, 2, range)

# Partition Data
set.seed(2)
train.index <- sample(nrow(mortgage.df), nrow(mortgage.df)*0.7)  
train.df <- mortgage.df[train.index, ]
valid.df <- mortgage.df[-train.index, ]

# initialize normalized training, validation data to originals
train.norm.df <- train.df
valid.norm.df <- valid.df

# use preProcess() from the caret package to normalize
norm.values <- preProcess(train.df[, c(1, 4, 15, 19, 20)], method = c("range"))
train.norm.df[, c(1, 4, 15, 19, 20)] <- predict(norm.values, train.df[, c(1, 4, 15, 19, 20)])
valid.norm.df[, c(1, 4, 15, 19, 20)] <- predict(norm.values, valid.df[, c(1, 4, 15, 19, 20)])

# Create formula for nn
col_names <- names(train.df)
form <- as.formula(paste("loan_status~",paste(col_names[!col_names %in% "loan_status"], collapse="+")))
form

# run nn with a single hidden layer with 1 node
nn <- neuralnet(form, data = train.norm.df, hidden = 1, linear.output = FALSE)
plot(nn)

training.prediction <- compute(nn, train.norm.df[,-c(21)])
training.class <- ifelse(training.prediction$net.result > 0.5,1,0)
train.pred1 <- confusionMatrix(as.factor(training.class), as.factor(train.norm.df$loan_status))
train.pred1

testing.prediction <- compute(nn, valid.norm.df[,-c(21)])
testing.class <- ifelse(testing.prediction$net.result > 0.5,1,0)
test.pred1 <- confusionMatrix(as.factor(testing.class), as.factor(valid.norm.df$loan_status))
test.pred1