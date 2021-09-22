### CH6 - Exercise 

###Q8
#A - use rnorm() function to gennerate a predictor X
#B - generate a vector Y 
#C - use best subset selection to select , use functionregsubsets()
#D - use forward and backward stepwise
#E - use lasso model, and use CV to a lamda
#F - Y = X^7...

#a 
set.seed(1)
X = rnorm(100)
eps = rnorm(100)
#b
beta0 = 3
beta1 = 2
beta2 = -3
beta3 = 0.3
Y = beta0 + beta1 * X + beta2 * X^2  + beta3 * X^3 + eps
#c
library(leaps)
data.full = data.frame(y = Y, x = X)
mod.full = regsubsets((y ~ poly(x,  10, raw = T), data = data.full, nvmax = 10)
mod.summary = summary(mod.full)
which.min(mod.summary$cp)
which.min(mod.summary$bic)
which.min(mod.summary$adjr2)
plot(mod.summary$cp, xlab = "Subset Size", ylab = "Cp", pch = 20, type = "l")
points(3, mod.summary$cp[3], pch = 4, col = "red", lwd = 5 )
plot(mod.summary$bic, xlab = "Subset Size", ylab = "BIC", pch = 20, type = "l")
points(3, mod.summary$bic[3], pch = 4, col = "red", lwd = 5 )
plot(mod.summary$adjr2, xlab = "Subset Size", ylab = "Adj_R^2", pch = 20, type = "l")
points(3, mod.summary$adjr2[3], pch = 4, col = "red", lwd = 5 )
coefficients(mod.full, id = 3)
#d
mod.fwd = regsubsets(y ~ poly(x, 10, raw = T), data = data.full, nvmax = 10, method = "forward")
mod.bwd = regsubsets(y ~ poly(x, 10, raw = T), data = data.full, nvmax = 10, method = "backward")
fwd.summary = summary(mod.fwd)
bwd.summary = summary(mod.bwd)
which.min(fwd.summary$bic)
which.min(bwd.summary$bic)
par(mfrow = c(3,2))
plot(fwd.summary$bic, xlab = "subsets size", ylab = "forward BIc", pch = 20, type = "l")
points(3, fwd.summary$bic[3], pch = 4, col ="red",lwd = 5 )
plot(bwd.summary$bic, xlab = "subsets size", ylab = "backward BIc", pch = 20, type = "l")
points(3, bwd.summary$bic[3], pch = 4, col ="red",lwd = 5 )
#maybe use cp and adjr2 too
#e
library(glmnet)
xmat = model.matrix(y ~ poly(x, 10, raw = T), data = data.full)[,-1]
mod.lasso = cv.glmnet(xmat, Y, alpha = 1)
best.lambda = mod.lasso$lambda.min
best.lambda
#0.0399
plot(mod.lasso)
#fit
best.model = glmnet(xmat, Y, alpha = 1)
predict(best.model, s = best.lambda, type = "coefficients")
#create a new Y with X^7
beta7 = 7
Y = beta0 + beta7 * X^7 +eps
data.full = data.frame(y = Y, x = X)
mod.full = regsubsets(y ~ poly(x, 10, raw = T), data = data.full, nvmax = 10)
mod.summary = summary(mod.full)
#find the model size
which.min(mod.summary$cp)
#2
which.min(mod.summary$bic)
#1
which.max(mod.summary$adjr2)
#4
coefficients(mod.full, id= 1)
#7
coefficients(mod.full, id = 2)
#2,7
coefficients(mod.full, id = 4)
#1, 2, 3, 7
#BIC works best here

xmat = model.matrix(y ~ poly(x, 10, raw = T), data = data.full)[, -1]
mod.lasso = cv.glmnet(xmat, Y, alpha = 1)
best.lambda = mod.lasso$lambda.min
best.lambda
best.model = glmnet (xmat, Y, alpha = 1)
predict(best.model, s = best.lambda, type = "coefficients")
#lasso picked the best varialbe but the intercept was quite off

###Q9 
#!!!!!unfinlished
#A split the data
#B linear least square
#C ridge reg
#D lasso reg
#E PCR
#F PLS
#G comment

#a 
library(ISLR)
set.seed(11)
sum(is.na(College))

train.size = dim(College)[1]/2
train = sample(1:dim(College)[1], train.size)
test = -train
College.train = College[train, ]
College.test = College[test, ]

#b 
lm.fit = lm(Apps~., data = College.train)
lm.pred = predict(lm.fit, College.test)
mean((College.test[, "Apps"] - lm.pred)^2)
#1026096

#=====this is diff? waiting to find out
#

#c 
library(glmnet)
train.mat = model.matrix(Apps~., data = College.train)
test.mat = model.matrix(Apps~., data = College.test)
grid = 10 ^ seq(4, -2, length = 100)
mod.ridge = cv.glmnet(train.mat, College.train[, "Apps"], 
alpha = 0, lambda = grid, thresh = 1e - 12)
lambda.best = mod.ridge$lambda.min
lambda.best
