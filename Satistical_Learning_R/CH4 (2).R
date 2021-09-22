require(ISLR)
require(MASS
        )
##linear discriminant analysis
lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=Year<2005)
lda.fit
plot(lda.fit)
Smarket.2005=subset(Smarket,Year==2005)
lda.pred=predict(lda.fit,Smarket.2005)
lda.pred[1:5,]                 
class(lda.pred)
data.frame(lda.pred$class,Smarket.2005$Direction)
data.frame(lda.pred)[1:5,]
mean(lda.pred$class==Smarket.2005$Direction)


## K-Nearest Neighbors
library(class)
?knn
attach(Smarket)
Xlag=cbind(Lag1,Lag2)
train=Year<2005
knn.pred=knn(Xlag[train,],Xlag[!train,],Direction[train],k=3)
table(knn.pred,Direction[!train])
mean(knn.pred==Direction[!train])
