###CH_4 Exercise
library(ISLR)
summary(Weekly)
pairs(Weekly)
cor(Weekly[,-9])
###Year and Volume appear to have a relationship
attach(Weekly)
glm.fit = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Weekly, family = binomial)
summary(glm.fit)

###Lag2 appears to be statistically significant
glm.probs = predict(glm.fit, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
table(glm.pred, Direction)


#the pr for correct:
(54+557)/(54+ 48 + 430 + 557)
#true up
577/(48+577)



#true down
54/(54+430)

###splited_data new model
train = (Year < 2009)
Weekly.0910 = Weekly[!train,]
glm.fit = glm(Direction ~ Lag2, data = Weekly, family = binomial, subset = train)
glm.probs = predict(glm.fit, Weekly.0910, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
Direction.0910 = Direction[!train]
table(glm.pred, Direction.0910)
mean(glm.pred == Direction.0910)

###LDA
library(MASS)
lda.fit = lda(Direction ~Lag2, data = Weekly, subset = train)
lda.pred = predict(lda.fit, Weekly.0910)
table(lda.pred$class, Direction.0910)
mean(lda.pred$class == Direction.0910)

###QDA
qda.fit = qda(Direction ~ Lag2, data = Weekly, subset = train)
qda.class = predict(qda.fit, Weekly.0910)$class
table(qda.class, Direction.0910)
mean(qda.class  == Direction.0910)

###KNN with k=1
library(class)
train.X = as.matrix(Lag2[train])
test.X = as.matrix(Lag2[!train])
train.Direction = Direction[train]
set.seed(1)
knn.pred = knn(train.X, test.X, train.Direction, k= 1)
table(knn.pred, Direction.0910)
mean(knn.pred == Direction.0910)

###conparison of models
#logistic regression and LDA methods provide the similar test error rates
#0.6, which is OK

###-------Q11-----------###

###a
library(ISLR)
summary(Auto)
attach(Auto)
mpg01= rep(0, length(mpg))
mpg01[mpg > median(mpg)] = 1
Auto = data.frame(Auto, mpg01)

###b plot the data to see the trend
cor(Auto[, -9])
pairs(Auto)
# we can see Anti-correlation with cylinders, weight, displacement, horsepower(with mpg)


###split into Tr and Te
train = (year%%2 == 0)
test = !train
Auto.train = Auto[train,]
Auto.test = Auto[test,]
mpg01.test = mpg01[test]


###LDA
library(MASS)
lda.fit = lda(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, subset = train)
lda.pred = predict(lda.fit, Auto.test )
mean(lda.pred$class != mpg01.test)
#TER=12.6%

###QDA
qda.fit = qda(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, subset = train)
qda.pred = predict(qda.fit, Auto.test)
mean(qda.pred$class != mpg01.test)
#TER=13.19%

###logistic 
glm.fit = glm(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, family = binomial, subset = train)
glm.probs = predict(glm.fit, Auto.test, type = "response")
glm.pred = rep(0, length(glm.probs))
glm.pred [glm.probs > 0.5] = 1
mean(glm.pred != mpg01.test)
#TER=12.09%

###KNN
library(class)
train.X = cbind(cylinders, weight, displacement, horsepower) [train, ]
test.X = cbind(cylinders, weight, displacement, horsepower) [test, ]
train.mpg01 = mpg01[train]
set.seed(1)
#k =1 
knn.pred = knn(train.X, test.X, train.mpg01, k= 1)
mean(knn.pred != mpg01.test)
#15.4%

#k =10 
knn.pred = knn(train.X, test.X, train.mpg01, k= 10)
mean(knn.pred != mpg01.test)
#16.5%

#k =100
knn.pred = knn(train.X, test.X, train.mpg01, k= 100)
mean(knn.pred != mpg01.test)
#14.3%
#so k = 100 seems to perform the best here

###Q12-----writing function ------jumped