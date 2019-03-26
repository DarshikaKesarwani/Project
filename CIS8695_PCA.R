rm(list = ls())
#setwd("C:/Users/lxue5/Dropbox/2019Sp/CIS8695/5 Dimension Reduction")
cereals.df <- read.csv("CSV_Cereals.csv") 

# Initial exploration
cereals.df<-cereals.df[,-c(1:3)]
cov(na.omit(cereals.df))
heatmap(cov(na.omit(cereals.df)), Rowv = NA, Colv = NA)

# compute PCs on two dimensions
pcs <- prcomp(data.frame(cereals.df$calories, cereals.df$protein)) 
summary(pcs) 
pcs$rot
scores <- pcs$x
head(scores, 5)

# compute PCs on more dimensions, omit the first 3 columns
pcs <- prcomp(na.omit(cereals.df[,-c(1:3)])) 
summary(pcs)
pcs$rot[,1:5]
scores <- pcs$x
head(scores, 5)

# Plot how cereals are different along different dimensions (PCs)
scores=data.frame("id"=seq(1,74),scores)
with(scores, plot(id,PC1, col="black"))
par(new=T)
with(scores, plot(id,PC4, col="red"))
# Plot Proportion of Variance 
PoV <- pcs$sdev^2/sum(pcs$sdev^2)
barplot(PoV, xlab = "Principal Components", ylab = "Proportion of Variance Explained", 
        names.arg = c(1:13))
# Simply plot the PC values (variance values)
plot(pcs)
#Create the biplot with principle components
biplot(pcs, col =c("red", "blue"))

# Normalization 
pcs.cor <- prcomp(na.omit(cereals.df), scale. = T)
summary(pcs.cor)
pcs.cor$rot[,1:5]
scores.cor <- pcs.cor$x
# Plot how cereals are different along different dimensions (PCs)
scores.cor=data.frame("id"=seq(1,74),scores.cor)
with(scores.cor, plot(id,PC1, col="black"))
par(new=T)
with(scores.cor, plot(id,PC13, col="red"))


# SVD
cereals.svd = svd(cereals.df)
plot(cereals.svd$d^2/sum(cereals.svd$d^2), type="l", xlab=" Singular vector", 
     ylab = "Variance explained")

# Reconstruct the data with only two singular vectors
cereals.recon = cereals.svd$u[,1:2] %*% diag(cereals.svd$d[1:2], 
              2, 2) %*% t(cereals.svd$v[,1:2])

# Compare with the original data
par(mfrow=c(1,2))
image(as.matrix(cereals.df), main="Cereal data Image")
image(cereals.recon,  main="Reconstructed Image")

# Compare with PCA
svd.m = svd(scale(cereals.df))
svd.m$v

pca.m = prcomp(cereals.df,scale=TRUE)
pca.m$rotation

