###CH10 - Exercise

###Q3 - K-means clustering
#a plot
#b random assign the cluster label to obs
#c compute the centroid for each cluster
#d assign each  to its closest centroid
#e repeat until no changing
#f color each cluster

set.seed(1)
x = cbind(c(1, 1, 0, 5, 6, 4), c(4, 3, 4, 1, 2, 0))
x

#a
plot(x[,1], x[,2])

#b
labels = sample(2, nrow(x), replace = T)
labels

#c
centroid1 = c(mean(x[labels == 1, 1]), mean(x[labels == 1, 2]))
centroid2 = c(mean(x[labels == 2, 1]), mean(x[labels == 2, 2]))
centroid1
centroid2

plot(x[, 1], x[, 2], col = (labels+1), pch = 20, cex =2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

#d
euclid = function(a, b){
  return(sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2))
}
assign_labels = function(x, centroid1, centroid2){
  labels = rep(NA, nrow(x))
  for (i in 1:nrow(x)){
    if (euclid(x[i,], centroid1) < euclid(x[i,], centroid2)){
      labels[i] = 1
    } else {
      labels[i] = 2
    }
  }   
  return(labels)
}
labels = assign_labels(x, centroid1, centroid2)
labels

#e
last_labels = rep( -1, 6)
while( !all(last_labels == labels)){
  last_labels = labels
  centroid1 = c(mean(x[labels == 1, 1]), mean(x[labels == 1, 2]))
  centroid2 = c(mean(x[labels == 2, 1]), mean(x[labels == 2, 2]))
  print(centroid1)
  print(centroid2)
  labels = assign_labels(x, centroid1, centroid2)
}
labels

plot(x[,1], x[,2], col = (labels + 1), pch = 20, cex = 2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)



###Q7
#show that, after standardized, 1 - correlation is proportional to the Euclidean distance
library(ISLR)
set.seed(1)
dsc = scale(USArrests)
a = dist(dsc)^2
b = as.dist( 1 - cor(t(dsc)))
summary(b/a)

###8
#compute PVE in two ways
#a prcomp() sdev output
#b compute PC loading first

#a
library(ISLR)
set.seed(1)

pr.out = prcomp(USArrests, center = T, scale = T)
pr.var = pr.out$sdev^2
pve = pr.var / sum(pr.var)
pve

#b  
loadings = pr.out$rotation
pve2 = rep(NA, 4)
dmean = apply(USArrests, 2, mean)
dsdev = sqrt(apply(USArrests, 2, var))
dsc = sweep(USArrests, MARGIN = 2, dmean, "-")
dsc = sweep(dsc, MARGIN = 2, dsdev, "/")
for (i in 1:4){
  proto_x = sweep(dsc, MARGIN = 2, loadings[,i], "*")
  pc_x = apply(proto_x, 1, sum)
  pve2[i] = sum(pc_x^2)
}
pve2 = pve2/sum(dsc^2)
pve2

#check gramma: spelling, marks, incompleteness


###Q9
#a use hierarchical clustering with complete linkage
#b cut into three clusters
#c do it after scaling
#d should we scale

library(ISLR)
set.seed(2)

#a
hc.complete = hclust(dist(USArrests), method = "complete")
plot(hc.complete)
cutree(hc.complete, 3)
table(cutree(hc.complete, 3))

#c
dsc = scale(USArrests)
hc.s.complete = hclust(dist(dsc), method = "complete")
plot(hc.s.complete)

#d
cutree(hc.s.complete, 3)
table(cutree(hc.s.complete, 3))
#it does affect the three outcome, so it is better to scaler it here, as the data were measured by diff_units

###Q10
#a generate a dataset: 3 classes, 20 obs each, with 50 variables
#b perform PCA , plot 1, 2 PC, make sure 3 classes seperate
#c K_means with K = 3, make comment
#d K = 2
#e K = 4
#f PC = 2, K =3
#g standradized dataset, K = 3

#a 
set.seed(1)
x = matrix(rnorm(20*3*50, mean = 0, sd = 0.001), ncol = 50)
x[1:20, 2] = 1
x[21:40, 1] = 2
x[21:40, 2] = 2
x[41:60, 1] = 1

#b
pca.out = prcomp(x)
summary(pca.out)
pca.out$x[, 1:2]
plot(pca.out$x[, 1:2], col = 2:4, xlab = "Z1", ylab = "Z2", pch = 19)

#c
km.out = kmeans(x, 3, nstart = 20)
table(km.out$cluster, c(rep(1, 20), rep(2, 20), rep(3, 20)))
#fit prefectly

#d
km.out = kmeans(x, 2, nstart = 20)
km.out$cluster
#two cluster merged

#e
km.out = kmeans(x, 4, nstart = 20)
km.out$cluster
#a cluster splited

#f
km.out = kmeans(pca.out$x[, 1:2], 3, nstart = 20)
table(km.out$cluster, c(rep(1, 20), rep(2, 20), rep(3, 20)))
#perfect match

#g
km.out = kmeans(scale(x), 3, nstart = 20)
km.out$cluster
#poorer resluts as standardizing affect the distance of data