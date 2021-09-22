###CH7 - Exercise

###Q6
#a polynomial reg
#b step function

#a poly reg CV=10
set.seed(1)
library(ISLR)
library(boot)
all.deltas = rep(NA, 10)
for (i in 1:10) {
  glm.fit = glm(wage ~ poly(age, i), data = Wage)
  all.deltas[i] = cv.glm(Wage, glm.fit, K = 10)$delta[2]
}
plot(1:10, all.deltas, xlab = "degree", ylab = "CV error", type  = "l", pch = 20, lwd = 2, ylim = c(1590, 1700))
min.point = min(all.deltas)
sd.points = sd(all.deltas)
abline (h = min.point + 0.2*sd.points, col = "red", lty = "dashed")
abline (h = min.point - 0.2*sd.points, col = "red", lty = "dashed")
legend("topright", " 0.2 - standard devaition lines", lty = "dashed", col = "red")
# d= 3 is reasonable 

# now the anova approach
fit.1 = lm(wage~poly(age, 1), data = Wage)
fit.2 = lm(wage~poly(age, 2), data = Wage)
fit.3 = lm(wage~poly(age, 3), data = Wage)
fit.4 = lm(wage~poly(age, 4), data = Wage)
fit.5 = lm(wage~poly(age, 5), data = Wage)
fit.6 = lm(wage~poly(age, 6), data = Wage)
fit.7 = lm(wage~poly(age, 7), data = Wage)
fit.8 = lm(wage~poly(age, 8), data = Wage)
fit.9 = lm(wage~poly(age, 9), data = Wage)
fit.10 = lm(wage~poly(age, 10), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5, fit.6, fit.7, fit.8, fit.9, fit.10)
# also d =3

#now plot it 
plot(wage~ age, data = Wage, col ="darkgrey")
agelims = range(Wage$age)
age.grid = seq(from = agelims[1], to = agelims[2])
lm.fit = lm(wage~poly(age, 3), data = Wage)
lm.pred = predict(lm.fit, data.frame(age = age.grid))
lines(age.grid, lm.pred, col = "blue", lwd =2)


#b step reg cutpoint = 1,...,10
all.cvs = rep(NA, 10)
for(i in 2:10){
  Wage$age.cut = cut(Wage$age, i)
  lm.fit = glm(wage~age.cut, data=Wage)
  all.cvs[i] = cv.glm(Wage, lm.fit, K =10 ) $delta[2]
  
}
plot(2:10, all.cvs[-1], xlab = "number of cuts", ylab = "CV error", type  = "l", pch = 20, lwd =2)
# cuts = 8
# now, train and plot
lm.fit = glm(wage~cut(age, 8), data = Wage)
agelims = range(Wage$age)
age.grid = seq(from = agelims[1], to = agelims[2])
lm.pred = predict(lm.fit, data.frame(age = age.grid))
plot(wage~age, data = Wage, col = "darkgrey")
lines(age.grid, lm.pred, col = "red", lwd = 2)



###Q7 look at marital and jobclass , explore appropriate model

library(ISLR)
set.seed(1)
summary(Wage$maritl)
par(mfrow = c(1, 2))
plot(Wage$maritl, Wage$age)
plot(Wage$jobclass, Wage$age)

# try polynomial and step function
fit = lm(wage ~ maritl, data = Wage)
deviance(fit)
#4858941
fit = lm(wage ~ jobclass, data = Wage)
deviance(fit)
#4998547
fit = lm(wage ~ maritl + jobclass, data = Wage)
deviance(fit)
#4654752

#splines 
#cannot be done because this is categorical variable

#GAMS
library(gam)
fit = gam(wage ~ maritl + jobclass + s(age, 4), data = Wage)
deviance(fit)
#4476501

# we can see that these two variables do add statistcally significance to the previously model


###Q8  -  explore auto dataset 
library(ISLR)
set.seed(1)
pairs(Auto)

#poly reg
rss = rep(NA, 10)
fits = list()
for( d in 1:10 ) {
  fits[[d]] = lm(mpg ~ poly(displacement, d), data = Auto)
  rss[d] = deviance(fits[[d]])
}
rss
# [1] 8378.822 7412.263 7392.322 7391.722 7380.838 7270.746 7089.716 6917.401 6737.801 6610.190

anova(fits[[1]], fits[[2]], fits[[3]], fits[[4]])
#quadratic is good here

library(glmnet)
library(boot)
cv.errs = rep (NA, 15)
for (d  in  1:15){
  fit = glm(mpg ~ poly(displacement, d), data = Auto)
  cv.errs[d] = cv.glm(Auto, fit, K =10)$delta[2]
}
which.min(cv.errs)
cv.errs  
#!!!??? cv CHOOSE 10_th degree ploy

#stepwise 
cv.errs = rep(NA, 10)
for (c in 2:10) {
  Auto$dis.cut = cut(Auto$displacement,c)
  fit = glm( mpg ~ dis.cut, data = Auto)
  cv.errs[c] = cv.glm( Auto, fit, K = 10)$delta[2]
  
}
which.min(cv.errs)
#[1], 9

#spline
library(splines)
cv.errs = rep(NA, 10)
for (df in 3:10) {
  fit = glm(mpg ~ ns(displacement, df = df), data = Auto)
  cv.errs[df] = cv.glm(Auto, fit, K = 10)$delta[2]
}
which.min(cv.errs)
cv.errs

#GAMS
library(gam)
fit = glm(mpg ~ s(displacement, 4) + s(horsepower, 4), data = Auto)
summary(fit)




###Q9
#a - cubic poly
#b - diff poly
#c - cv choose poly
#d - use bs() to fit reg s
#e - with diff df
#f - cv choose df

set.seed(1)
library(MASS)
attach(Boston)
#a
lm.fit = lm(nox ~ poly(dis, 3), data = Boston)
summary(lm.fit)
dislim = range(dis)
dis.grid = seq(from = dislim[1], to = dislim[2], by = 0.1)
lm.pred = predict(lm.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "darkgrey")
lines(dis.grid, lm.pred, col = "red", lwd = 2)
#seems to fit it well

#b
all.rss = rep(NA, 10)
for (i in 1:10) {
  lm.fit = lm(nox ~ poly(dis, i), data = Boston)
  all.rss[i] = sum(lm.fit$residuals^2)
}
all.rss
#train RSS monotonically decreases

#c - using a 10_CV
library(boot)
all.deltas = rep(NA, 10)
for ( i in 1:10) {
  glm.fit = glm (nox ~ poly(dis, i), data = Boston)
  all.deltas[i] = cv.glm(Boston, glm.fit, K =10)$delta[2]
}
plot(1:10, all.deltas, xlab = "Degree", ylab = "CV error", type = "l", pch = 20, lwd = 2)
# d= 4 as a best

#d 
# As we split the range of 'dis' into four range at [4, 7, 11]
library(splines)
sp.fit = lm(nox ~ bs(dis, df =4, knots = c(4, 7, 11)), data = Boston)
summary(sp.fit)
sp.pred = predict(sp.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "darkgrey")
lines(dis.grid, sp.pred, col ="red", lwd = 2)
#it fits well except the extreme

#e
all.cv = rep(NA, 16)
for (i in 3:16){
  lm.fit = lm(nox ~bs(dis, df = i), data = Boston)
  all.cv[i] = sum(lm.fit$residuals^2)
}
all.cv[ -c(1, 2)]
#train RSS dereases til df = 14

#f
#10_fold_CV

all.cv = rep(NA, 16)
for (i in 3:16){
  lm.fit = glm(nox ~ bs(dis, df = i), data = Boston)
  all.cv[i] = cv.glm(Boston, lm.fit, K = 10)$delta[2]
}
plot(3:16, all.cv[ - c(1, 2)], lwd = 2, type = "l", xlab = "df", ylab = "CV error")
# it is more jumpy but it attains minimum at df = 10

###10
#a - use forward stepwise to ...
#b - fit GAM
#c - evaluate the model
#d - non_linear?

#a
set.seed(1)
library(ISLR)
library(leaps)
attach(College)
train = sample ( length(Outstate), length(Outstate)/2)
test = - train
College.train = College[train, ]
College.test = College[test, ]
reg.fit = regsubsets(Outstate ~., data = College.train, nvmax = 17, method = "forward")
reg.summary = summary(reg.fit)
par(mfrow = c(1, 3))
plot(reg.summary$cp, xlab = "Number of variables", ylab = "Cp", type = "l")
min.cp = min(reg.summary$cp)
std.cp = sd(reg.summary$cp)
abline(h = min.cp + 0.2*std.cp, col = "red", lty = 2)
abline(h = min.cp - 0.2*std.cp, col = "red", lty = 2)
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
min.bic = min(reg.summary$bic)
std.bic = sd(reg.summary$bic)
abline(h = min.bic + 0.2*std.bic, col = "red", lty = 2)
abline(h = min.bic - 0.2*std.bic, col = "red", lty = 2)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l", ylim = c(0.4, 0.84))
max.adjr2 = max(reg.summary$adjr2)
std.adjr2 = sd(reg.summary$adjr2)
abline(h = max.adjr2 + 0.2*std.adjr2, col = "red", lty = 2)
abline(h = max.adjr2 - 0.2*std.adjr2, col = "red", lty = 2)
#size = 6 should be good
reg.fit = regsubsets(Outstate ~., data = College, method = "forward")
coefi = coef(reg.fit, id = 6)
names(coefi)
# we have "(Intercept)" "PrivateYes"  "Room.Board"  "PhD"         "perc.alumni" "Expend"      "Grad.Rate"  

#b
library(gam)
gam.fit = gam(Outstate ~ Private + s(Room.Board, df = 2)+ s(PhD, df = 2) + 
                s(perc.alumni, df = 2) +  s(Expend, df = 5) + s(Grad.Rate, df = 2), data = College.train)
par(mfrow = c(2,3))              
plot(gam.fit, se =T, col = "blue")

#c 
gam.pred = predict(gam.fit, College.test)
gam.err = mean((College.test$Outstate - gam.pred)^2)
gam.err
#3349290
gam.tss = mean((College.test$Outstate - mean(College.test$Outstate))^2)
test.rss = 1 - gam.err/gam.tss
test.rss
#0.766
