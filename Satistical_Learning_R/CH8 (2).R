###CH8  Exrecise

###7

#we will try ntree from 1 to 500, amd mtry of p, p/2,p^0.5
#in Boston data, p = 13

library(MASS)
library(randomForest)
set.seed(1101)

#train and test
train = sample(dim(Boston)[1], dim(Boston)[1]/2)
X.train = Boston[train, -14]               
X.test = Boston[ - train, -14]  
Y.train = Boston[train, 14]               
Y.test = Boston[ - train, 14]  

p = dim(Boston)[2] - 1
p.2 = p/2
p.sq = sqrt(p)

rf.boston.p = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test,
                          mtry = p, ntree = 500)
rf.boston.p.2 = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test,
                           mtry = p.2, ntree = 500)
rf.boston.p.sq = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test,
                           mtry = p.sq, ntree = 500)
#cut and paste is not a good habit
plot(1:500, rf.boston.p$test$mse, col = "green", type = "l", xlab = "Number of trees", 
     ylab = "Test MSE", ylim = c(10, 19))
lines(1:500, rf.boston.p.2$test$mse, col = "red", type = "l")
lines(1:500, rf.boston.p.sq$test$mse, col = "blue", type = "l")
legend("topright", c("m = p", "m = p/2", "m = sqrt(p)"), col = c("green", "red", "blue"), cex = 1, lty = 1)


###Q8 ------- there is a problem with the dataset
#predict sales using reg tree
#a - split
#b - reg tree to trainset
#c - cv to choose prun
#d - use bagging
#e - use random forest

#a
library(ISLR)
attach(Carseats)
set.seed(1)

train = sample(dim(Carseats)[1], dim(Carseats)[1]/2)
Carseats.train = Carseats[train, ]
Carseats.test = Carseats[-train, ]

#b
library(tree)
tree.carseats = tree(Sales ~ ., data = Carseats.train)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty = 0)
pred.carseats = predict(tree.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.carseats)^2)
## [1] 4.149

#c
cv.carseats = cv.tree(tree.carseats, FUN = prune.tree)
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")
# Best size = 9
pruned.carseats = prune.tree(tree.carseats, best = 9)
par(mfrow = c(1, 1))
plot(pruned.carseats)
text(pruned.carseats, pretty = 0)

#d
library(randomForest)
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
bag.carseats = randomForest(Sales ~ ., data = Carseats.train, mtry = 10, ntree = 500, 
                            importance = T)
bag.pred = predict(bag.carseats, Carseats.test)
mean((Carseats.test$Sales - bag.pred)^2)
## [1] 2.586
importance(bag.carseats)
##             %IncMSE IncNodePurity
## CompPrice   13.8790       131.095
## Income       5.6042        77.033
## Advertising 14.1720       129.218
## Population   0.6071        65.196
## Price       53.6119       506.604
## ShelveLoc   44.0311       323.189
## Age         20.1751       189.269
## Education    1.4244        41.811
## Urban       -1.9640         8.124
## US           5.6989        14.307
#Bagging improves the test MSE to 2.58. We also see that Price, ShelveLoc and Age are three most important predictors of Sale.

#e
rf.carseats = randomForest(Sales ~ ., data = Carseats.train, mtry = 5, ntree = 500, 
                           importance = T)
rf.pred = predict(rf.carseats, Carseats.test)
mean((Carseats.test$Sales - rf.pred)^2)
## [1] 2.87
importance(rf.carseats)
##             %IncMSE IncNodePurity
## CompPrice   11.2746        126.64
## Income       4.4397        101.63
## Advertising 12.9346        137.96
## Population   0.2725         78.78
## Price       49.2418        449.52
## ShelveLoc   38.8406        283.46
## Age         19.1329        195.14
## Education    1.9818         54.26
## Urban       -2.2083         11.35
## US           6.6487         26.71
#In this case, random forest worsens the MSE on test set to 2.87. Changing m varies test MSE between 2.6 to 3. We again see that Price, ShelveLoc and Age are three most important predictors of Sale.


###Q9


#a - split
#b - use purchase as the response
#c - type the name of the tree
#d - plot 
#e - predict the test data and produce a confusion matrix
#f - cv.tree
#g - plot CV_error vs. tree size
#h - the best size
#i - produce the choose tree
#j - compare the Tr
#k - compare the Te

#a
library(ISLR)
attach(OJ)
set.seed(1013)

trian = sample(dim(OJ)[1], 800)
OJ.train = OJ[train, ]
OJ.test = OJ[- train, ]

#b
library(tree)
oj.tree = tree(Purchase ~., data = OJ.train)
summary(oj.tree)
#it use 7 variables and has 15 nodes

#c
oj.tree

#d 
plot(oj.tree)
text(oj.tree, pretty = 0)
#it seems like LoyalCH is the most improtant variable, followed by PriceDiff

#e
oj.pred = predict(oj.tree, OJ.test, type = "class")
table(OJ.test$Purchase, oj.pred)

#f
cv.oj = cv.tree(oj.tree, FUN = prune.tree)

#g
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

#h 
#here size = 3

#i
oj.pruned = prune.tree(oj.tree, best = 3)

#j
summary(oj.pruned)
#misclassi error rate : 0.2292 , it was around 0.1

#k
pred.unpruned = predict(oj.tree, OJ.test, type = "class")
misclass.unpruned = sum(OJ.test$Purchase != pred.unpruned)
misclass.unpruned/length(pred.unpruned)
#[1] 0.2129743
pred.pruned = predict(oj.pruned, OJ.test, type = "class")
misclass.pruned = sum(OJ.test$Purchase != pred.pruned)
misclass.pruned/length(pred.pruned)
#[1] 0.2337821
#not too different


###Q10

#a - preprocess the data : remove and log_transform
#b - split
#c - use boosting, with 1000 trees and a range of lambda,plot the training
#d - plot Te
#e - compare the boosting Te to Reg method
#f - which var
#g - apply bagging, report Te

#a
library(ISLR)
sum(is.na(Hitters$Salary))
Hitters = Hitters[ - which(is.na(Hitters$Salary)),]
sum(is.na(Hitters$Salary))
Hitters$Salary = log(Hitters$Salary)

#b
train = 1:200
Hitters.train = Hitters[train, ]
Hitters.test = Hitters[ - train, ]

#c
library(gbm)
  
  set.seed(103)
  pows = seq(-10, -0.2, by = 0.1)
  lambdas = 10^pows
  length.lambdas = length(lambdas)
  train.errors = rep(NA, length.lambdas)
  test.errors = rep(NA, length.lambdas)
  for (i in 1:length.lambdas) {
    boost.hitters = gbm(Salary ~ ., data = Hitters.train,  distribution = "guassian",
                        n.trees = 1000, shrinkage = lambdas[i])
    train.pred = predict(boost.hitters, Hitters.train, n.trees = 1000)
    test.pred = predict(boost.hitters, Hitters.test, n.trees = 1000)
    train.errors[i] = mean((Hitters.train$Salary - train.pred)^2)
    test.errors[i] = mean((Hitters.test$Salary - test.pred)^2)
  }
  plot(lambdas, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", 
       col = "blue", pch = 20)
# Error in gbm(Salary ~ ., data = Hitters.train, distribution = "guassian",  : 
# Distribution guassian is not supported.

#d 
plot(lambdas, test.errors, type = "b", xlab = "Shrinkage", ylab = "Test MSE", 
     col = "red", pch = 20)
