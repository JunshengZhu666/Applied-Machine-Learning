### CH9 - SVM

###Q4
#we create a random dataset which lies along y = 3*x^2 + 4
set.seed(1)
x = rnorm(100)
y = 3 *x^2 + 4 + rnorm(100)
train = sample(100, 50)
y[train] = y[train] + 3
y[-train] = y[ - train ] - 3
#plot 
plot(x[train], y[train], pch = "+", lwd = 4, col = "red", ylim = c(-4, 20)
     ,xlab = "X", ylab = "Y")
points(x[-train], y[-train], pch = "o", lwd = 4, col = "blue")

#the plot showed a nonlinear patten, we now create both train
#and test dataset by taking half of the + - classes
#and z vector of 0 and 1

set.seed(315)
z = rep(0, 100)
z[train] = 1
#take each 25 from + - 
final.train = c(sample(train, 25), sample(setdiff(1 : 100, train),25))
data.train = data.frame(x = x[final.train], y = y[final.train], 
        z = as.factor(z[final.train]))    
data.test = data.frame(x = x[ - final.train], y = y[ - final.train], 
        z = as.factor(z[ - final.train]))       
library(e1071)
svm.linear = svm(z~., data = data.train, kernel = "linear", cost = 10)
plot(svm.linear, data.train)
table(z[final.train], predict(svm.linear, data.train))
#it made 10 errors

#now try poly ssssssvm
set.seed(32545)
svm.poly = svm(z~., data = data.train, kernel = "polynomial", cost = 10)
plot(svm.poly, data.train)
table(z[final.train], predict(svm.poly, data.train))
#it made 9 errors

#try radial kernel
set.seed(996)
svm.radial = svm(z~., data = data.train, kernel = "radial", gamma = 1, cost = 10)
plot(svm.radial, data.train)
table(z[final.train], predict(svm.radial, data.train))
#0 error
#for test
plot(svm.linear, data.test)
#1
plot(svm.poly, data.test)
#lots of
plot(svm.radial, data.test)
#0


###Q5
#generate a dataset n = 500, p = 2, with quadratic relation
#A 
#B plot obs
#C linear logistic
#D predict and plot
#E nonlinear logistic
#F predict and plot
#G use linear support vector classifier
#H use nonlinear kernel

set.seed(421)
x1 = runif(500) - 0.5
x2 = runif(500) - 0.5
y = 1 *(x1^2 - x2^2 > 0)

#b
plot(x1[y == 0], x2[y == 0], col = "red", xlab = "X1", ylab = "X2",
     pch = "+")
points(x1[y == 1], x2[y == 1], col = "blue", pch = 4)

#c 
lm.fit = glm( y ~ x1 + x2, family = binomial)
summary(lm.fit)

#d
data = data.frame(x1 = x1, x2 = x2, y = y)
lm.prob = predict(lm.fit, data, type = "response")
lm.pred = ifelse(lm.prob > 0.52, 1, 0)
data.pos = data[lm.pred == 1,]
data.neg = data[lm.pred == 0,]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2",
     pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)
#we can see a linear boundry

#e
lm.fit = glm(y ~ poly(x1, 2) + poly(x2, 2) + I(x1 *x2), 
             data = data, family = binomial)
lm.prob = predict(lm.fit, data, type = "response")
lm.pred = ifelse(lm.prob > 0.5, 1, 0)
data.pos = data[lm.pred == 1, ]
data.neg = data[lm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)
# it is a close apporch

#g
library(e1071)
svm.fit = svm(as.factor(y) ~ x1 + x2 , data, kernel = "linear",
              cost = 0.1)
svm.pred = predict(svm.fit, data)
data.pos = data[svm.pred == 1, ]
data.neg = data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)
#failed to class a single one

#h
svm.fit = svm(as.factor(y) ~ x1 + x2 , data, gamma = 1)
svm.pred = predict(svm.fit, data)
data.pos = data[svm.pred == 1, ]
data.neg = data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)
# a colse one

#i
#both logistic and SV linear performed bad,
#while quadratic terms performed well,
#and radial kernel is good without tuning too many parameters