rm(list = ls())

install.packages(c("OpenImageR","ClusterR"))
install.packages("OpenImageR")
library(OpenImageR)
library(ClusterR)

img = readImage("frog.jpg")
img_resize = resizeImage(img, 350, 350, method = 'bilinear') 
imageShow(img_resize) 

img_vector = apply(img_resize, 3, as.vector)                                # vectorize RGB
dim(img_vector)

km_mb = MiniBatchKmeans(img_vector, clusters = 5, batch_size = 20, num_init = 5, max_iters = 100, 
                        init_fraction = 0.2, initializer = 'kmeans++', early_stop_iter = 10,
                        verbose = F)
pr_mb = predict_MBatchKMeans(img_vector, km_mb$centroids)

getcent_mb = km_mb$centroids

new_im_mb = getcent_mb[pr_mb, ]
dim(new_im_mb) = c(nrow(img_resize), ncol(img_resize), 3)

imageShow(new_im_mb)

