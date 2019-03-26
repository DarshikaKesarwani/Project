rm(list = ls())
#setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/3 NB SVM KNN")

options(repos = c(CRAN = "http://cran.rstudio.com"))
install.packages("sentimentr")
install.packages("tm")
library(sentimentr)
library(tm)
library(e1071)
library(caret)

data_insults <- read.csv("kaggle_insults_1.csv",header=TRUE)

insult_comment<-as.vector(data_insults[,"Comment"])
insult_score<-as.vector(data_insults[,"Insult"])

clean.data = function(text) {
  
  # delete @ word and remove possible re-tweet entries
  text = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", text)
  text = gsub("@\\w+", "", text)
  # delete punctuation and remove digits 0-9
  text = gsub("[[:punct:]]", "", text )
  text = gsub("[[:digit:]]", "", text)
  # delete html links and unnecessary tabs,spaces
  text = gsub("http\\w+", "", text)
  text = gsub("[ \t]{2,}", "", text)
  text = gsub("^\\s+|\\s+$", "", text)
  
  return(text)
}

# Clean text
insult_comment<-clean.data(insult_comment)

handle.error = function(x)
{
  # create missing value
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error=function(e) e)
  # if not an error
  if (!inherits(try_error, "error"))
    y = tolower(x)
  # result
  return(y)
}

# Turn to lower-case
insult_comment = sapply(insult_comment, handle.error)

# remove NAs and nullifying the column names
insult_comment = insult_comment[!is.na(insult_comment)]
insult_comment[1]
names(insult_comment) = NULL


# builds a document term matrix
insult_corpus <- Corpus(VectorSource(insult_comment))
insult_dtm <- DocumentTermMatrix(insult_corpus)
dim(insult_dtm)

# removing words that do not occur in at least 1 percent of those documents
insult_dtm <- removeSparseTerms(x = insult_dtm, sparse = 0.99)
dim(insult_dtm)
inspect(insult_dtm[1:10,1:20])

# Convert all entries to binary
insult_dtm <- weightBin(insult_dtm)


# Create training and validation data
insult_df <- as.data.frame(as.matrix(insult_dtm))
insult_score <- factor(ifelse(insult_score == 1, "Insulting", "Non-Insulting"))
set.seed(123456)
insult_sampling_vector <- createDataPartition(insult_score, p = 0.70, list = FALSE)
insult_df_train <- insult_df[insult_sampling_vector,]
insult_df_valid <- insult_df[-insult_sampling_vector,]
insult_score_train = insult_score[insult_sampling_vector]
insult_score_valid = insult_score[-insult_sampling_vector]


# Naive Bayes Model
nb_insult <- naiveBayes(insult_df_train, insult_score_train)

# Prediction: Training
insult_train_pred <- predict(nb_insult, insult_df_train)
confusionMatrix(insult_train_pred, insult_score_train)

# Prediction: Validation
insult_valid_pred <- predict(nb_insult, insult_df_valid)
confusionMatrix(insult_valid_pred, insult_score_valid)





