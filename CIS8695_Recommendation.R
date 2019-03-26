rm(list = ls())
setwd("/Users/darsh/Documents/R")

#Collaborative filtering
install.packages("recommenderlab")
library(recommenderlab)
library("ggplot2")
data(MovieLense)

ratings_movies <- MovieLense[rowCounts(MovieLense) > 50,
                               colCounts(MovieLense) > 100]

# Explore 
hist(getRatings(ratings_movies), breaks="FD")
image(ratings_movies[1:5,1:5])
rating_df<-as(ratings_movies, "data.frame")
head(rating_df)


which_train <- sample(x = c(TRUE, FALSE), size = nrow(ratings_movies),
                        replace = TRUE, prob = c(0.8, 0.2))

recc_data_train <- ratings_movies[which_train, ]
recc_data_test <- ratings_movies[!which_train, ]

## Item Based Recommendation
ibcf_recc_model <- Recommender(data = recc_data_train, method = "IBCF", parameter = list(k = 30))
ibcf_model_details <- getModel(ibcf_recc_model)

## Explore similarity between movies
dim(ibcf_model_details$sim)   # Similarity matrix dimension
image(ibcf_model_details$sim[1:20, 1:20], main = "Heatmap of the first rows and columns")
# Overall distribution of similarity
col_sums <- colSums(ibcf_model_details$sim > 0)
qplot(col_sums) + stat_bin(binwidth = 1) + ggtitle("Distribution of the column count")
# Which movie is similar to most of others
which_max <- order(col_sums, decreasing = TRUE)[1:6]
rownames(ibcf_model_details$sim)[which_max]


## Make recommendations
n_recommended <- 6
ibcf_recc_predicted <- predict(object = ibcf_recc_model, newdata = recc_data_test, n = n_recommended)
ibcf_recc_predicted
slotNames(ibcf_recc_predicted)

# Recommendations for the first user
ibcf_recc_predicted@items[[1]]
recc_user_1 <- ibcf_recc_predicted@items[[1]]
movies_for_user_1 <- ibcf_recc_predicted@itemLabels[recc_user_1]
movies_for_user_1

# Generate a matrix format
ibcf_recc_matrix <- sapply(ibcf_recc_predicted@items, 
                           function(x){colnames(ratings_movies)[x]})
View(ibcf_recc_matrix[, 1:4])

# Plot distribution of # of recommendations for each movie
number_of_items <- factor(table(ibcf_recc_matrix))
chart_title <- "Distribution of the number of items for IBCF"
qplot(number_of_items) + ggtitle(chart_title)

# Which are most recommended movies?
number_of_items_sorted <- sort(number_of_items, decreasing = TRUE)
number_of_items_top <- head(number_of_items_sorted, n = 4)
table_top <- data.frame(number_of_items_top)
table_top


## User Based Recommendation
ubcf_recc_model <- Recommender(data = recc_data_train, method = "UBCF")
ubcf_model_details <- getModel(ubcf_recc_model)
ubcf_model_details

ubcf_model_details$data


n_recommended <- 5
ubcf_recc_predicted <- predict(object = ubcf_recc_model, 
                               newdata = recc_data_test, n = n_recommended)
ubcf_recc_predicted

ubcf_recc_matrix <- sapply(ubcf_recc_predicted@items, 
                           function(x){ colnames(ratings_movies)[x] })
dim(ubcf_recc_matrix)
View(ubcf_recc_matrix[, 1:4])

# Plot 
number_of_items <- factor(table(ubcf_recc_matrix))
chart_title <- "Distribution of the number of items for UBCF"
qplot(number_of_items) + ggtitle(chart_title)

# Which are most recommended movies?
number_of_items_sorted <- sort(number_of_items, decreasing = TRUE)
number_of_items_top <- head(number_of_items_sorted, n = 4)
table_top <- data.frame(number_of_items_top)
table_top



image(MovieLense, main = "Heatmap of the rating matrix")

min_n_movies <- quantile(rowCounts(MovieLense), 0.99)
min_n_users <- quantile(colCounts(MovieLense), 0.99)
image(MovieLense[rowCounts(MovieLense) > min_n_movies,colCounts(MovieLense) > min_n_users], main ="Heatmap of the top users and movies")

ratings_movies <- MovieLense[rowCounts(MovieLense) > 50,colCounts(MovieLense) > 100]
min_movies <- quantile(rowCounts(ratings_movies), 0.98)

min_users <- quantile(colCounts(ratings_movies), 0.98)
image(ratings_movies[rowCounts(ratings_movies) > min_movies,colCounts(ratings_movies) > min_users], main = "Heatmap of the top users and movies")


## Binarize method, item-based
ratings_movies_viewed <- binarize(ratings_movies, minRating = 1)

which_train <- sample(x = c(TRUE, FALSE), size = nrow(ratings_movies_viewed),
                      replace = TRUE, prob = c(0.8, 0.2))
recc_data_train <- ratings_movies_viewed [which_train, ]
recc_data_test <- ratings_movies_viewed [!which_train, ]

recc_model <- Recommender(data = recc_data_train, method = "IBCF",
                          parameter = list(method = "Jaccard"))
model_details <- getModel(recc_model)


n_recommended <- 6
recc_predicted <- predict(object = recc_model, 
                          newdata = recc_data_test, n = n_recommended)
recc_matrix <- sapply(recc_predicted@items, function(x){
  colnames(ratings_movies)[x]
})

View(recc_matrix[, 1:4])




## Content based Filtering
# download the movieTitle data
movie_URL <- "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movieTitleDF <- read.table(movie_URL, header = F, sep = "|", quote = "\"")

# rename the column names of both the datasets and remove the unwanted columns
names(movieTitleDF) <- c("MovieID", "Title", "ReleaseDate", "VideoReleaseDate", "IMDB", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western")
movieTitleDF$ReleaseDate <- NULL; 
movieTitleDF$VideoReleaseDate <- NULL
movieTitleDF$IMDB <- NULL
movieTitleDF <- unique(movieTitleDF)
str(movieTitleDF)

# download the user-rating data
users_URL <- "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
userDF <- read.table(users_URL, header = F, sep = "\t", quote = "\"")
names(userDF) <- c("UserID", "ItemID", "Rating")
userDF <- userDF[,1:3]
str(userDF)


# Cluster the movies based on their genre affiliation using k-means
clusterMovies<-function(movieTitleDF){
  set.seed(123)
  i<-1
  #get rid of movie ids and titles
  movieTitleDF<-movieTitleDF[,c(-1,-2)]
  movieCluster <- kmeans(movieTitleDF, 10, nstart = 20)
  return(movieCluster)
}

# find all the movies already watched and rated by the selected user
getUserInfo<-function(dat,id){
  a<-subset(dat, UserID==id,select=c(ItemID, Rating))
  # allocate 0 to the cluster column
  cluster<-0
  activeUser <- data.frame( a[order(a$ItemID),] ,cluster)
  return(activeUser)
}


# assign cluster numbers to each movie
setUserMovieCluster<-function(movieCluster, activeUser){
  df1<- data.frame(cbind(movieTitleDF$MovieID, clusterNum = movieCluster$cluster))
  names(df1)<-c("movie_id", "cluster")
  activeUser$cluster<-df1[match(activeUser$ItemID, df1$movie_id),2]
  return(activeUser)
}


# calculate the average movie rating for each cluster in user profile
# focus on the clusters that the user rates high
getAverageClusterRating<-function(movieCluster, activeUser){
  like<-aggregate(activeUser$Rating, by=list(cluster=activeUser$cluster), mean)
  if(max(like$x)<3){
    like<-as.vector(0)
   } else{
    like<-as.vector(t(max(subset(like, x>=3, select=cluster))))
  }
  return(like)
}


# select movies from a cluster
getGoodMovies<-function(like, movieCluster, movieTitleDF){
  df1<- data.frame(cbind(movieTitleDF$MovieID, clusterNum = movieCluster$cluster))
  names(df1)<-c("movie_id", "cluster")
  if(like==0){
    recommend<-movieTitleDF[sample.int(n = dim(titleFilmDF)[1], size = 100), 1]
  }
  else{
    recommend<-as.vector(t(subset(df1, cluster==like, select=movie_id)))
  }
  return(recommend)
}


# find all the movies that the user has not yet seen
getRecommendedMovies<-function(movieTitleDF, userDF, userid){
  movieCluster<-clusterMovies(movieTitleDF)
  activeUser<-getUserInfo(userDF, userid)
  activeUser<-setUserMovieCluster(movieCluster, activeUser)
  like<-getAverageClusterRating(movieCluster, activeUser)
  recommend<-getGoodMovies(like, movieCluster, movieTitleDF)
  # only select not yet watched movies
  recommend<-recommend[-activeUser$ItemID]
  mov_title<-movieTitleDF[match(recommend,movieTitleDF$MovieID),2]
  recommend<-data.frame(recommend,mov_title)
  return(recommend)
}


# suggest a certain number of movies to a particular user
suggestMovies<-function(movieTitleDF, userDF, userid, num_movies){
  #get suggestions
  suggestions = getRecommendedMovies(movieTitleDF, userDF, userid)
  #select stated number of selections
  suggestions = suggestions[1:num_movies,]
  writeLines("You may also like these movies:")
  #print suggestions without column headers or row indices
  write.table(suggestions[2], row.names = FALSE, col.names = FALSE)
}


# choose any UserID from the user dataframe (userDF) and use the suggestMovies() 
# to find some (e.g., 5) recommended movies for the user
suggestMovies(movieTitleDF, userDF, 196, 5)


# Hybrid Recommender
data("MovieLense")
MovieLense50 <- MovieLense[rowCounts(MovieLense) >50,]
train <- MovieLense50[1:100]
test <- MovieLense50[101:105]

hybrid_recom <- HybridRecommender(
  Recommender(train, method = "UBCF"),
  Recommender(train, method = "RANDOM"),
  weights = c(.7,.3)
)

hybrid_recom

getModel(hybrid_recom)

as(predict(hybrid_recom, test, n=10), "list")




