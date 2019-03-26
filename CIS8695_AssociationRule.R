rm(list = ls())
setwd("/Users/darsh/Documents/R")

install.packages("arules")
library(arules)
fp.df <- read.csv("CSV_Faceplate.csv")

# remove first column and convert to matrix
fp.mat <- as.matrix(fp.df[, -1])

# convert the binary incidence matrix into a transactions database
fp.trans <- as(fp.mat, "transactions")

# Explore data
inspect(fp.trans)
summary(fp.trans)
image(fp.trans)
itemFrequencyPlot(fp.trans)


## get rules
# when running apriori(), include the minimum support, minimum confidence, and target
# as arguments. 
rules <- apriori(fp.trans, parameter = list(supp = 0.2, conf = 0.5, target = "rules"))

# inspect the first six rules, sorted by their lift
inspect(head(sort(rules, by = "lift")))


rules.tbl <- inspect(rules)
rules.tbl[rules.tbl$support >= 0.04 & rules.tbl$confidence >= 0.7,]




book.df <- read.csv("CSV_CharlesBookClub.csv")
book.df <- book.df[, 8:18]
book.df <- ifelse(book.df>0,1,0)
book.df <- as.matrix(book.df, df[,-1])

book.trans <- as(book.df, "transactions")
inspect(book.trans)

book.rules <- apriori(book.trans, parameter = list(supp = 0.1, conf = 0.5, target = "rules"))

inspect(sort(book.rules, by = "lift"))

book.rules.tbl <- inspect(book.rules)
book.rules.tbl[book.rules.tbl$support >= 0.04 & book.rules.tbl$confidence >= 0.3,]



