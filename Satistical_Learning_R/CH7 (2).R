#Nonlinear Models

require(ISLR)
attach(Wage)

###Ploynomials 4th
fit=lm(wage~poly(age,4),data=Wage)
summary(fit)

#poly use orthogonal
#NOW plot
agelims=range(age)
age.grid=seq(from=agelims[1], to=agelims[2])
age.grid
preds= predict(fit, newdata = list(age=age.grid),se=TRUE)
se.bands=cbind(preds$fit + 2*preds$se, preds$fit - 2*preds$se)
plot(age, wage, col = "darkgrey")
lines(age.grid, preds$fit, lwd=2, col="blue")
matlines(age.grid, se.bands, col="blue", lty= 2)

#other way, direct
fita = lm(wage ~ age + I(age^2) + I(age^3)+ I(age^4), data = Wage)
summary(fita)
#here I()is a 'wrapper' function
plot(fitted(fit), fitted(fita))
#by using orthogonal ploy, we can test coeffi separately
#this is a nested sequence
#anova way()

###ploy nonimal logistic reg
#to code the big earners('>250k'")
fit=glm(I(wage>250) ~ poly(age,3), data = Wage, family= binomial)
summary((fit))
preds=predict(fit, list(age= age.grid), se = T)
se.bands = preds$fit + cbind(fit=0, lower= -2*preds$se, upper = 2*preds$se)
se.bands[1:5,]
# use logist tranformation to our 
probs.bands=exp(se.bands)/(1+exp(se.bands))
#...

###Splines
#cubic splines
require(splines)
fit=lm(wage~bs(age, knots=c(25,40,60)), data =Wage)
plot(age, wage, col="darkgrey")
lines(age.grid, predict(fit, list(age=age.grid)), col="darkgreen", lwd =2)
abline(v=c(25, 40, 60), lty=2, col="darkgreen")

#splines does not require knot selection, but we can have smoothing parameters
#by choosing effective d.f.
fit= smooth.spline(age, wage, df=16)
lines(fit, col="red", lwd = 2)
fit= smooth.spline(age, wage, df=4)
lines(fit, col="blue", lwd = 2)
#we can use LOOCV to select the smoothing parameters
fit=smooth.spline(age, wage, cv= TRUE)
lines(fit, col="purple", lwd=2)
fit



###Generalized Additive Models
# we can have multiple nonlinear terms 
#using GAM
require(gam)
gam1=gam(wage~s(age, df=4) + s(year, df=4) + education, data= Wage)
#ar(mfrow=C(1,3))
plot(gam1, se=T)

#to see if we need a linear terms for year
gam2a=gam(I(wage>250)~s(age, df=4) +year + education, data = Wage, family = binomial)
gam2=gam(I(wage>250)~s(age, df=4)  + education, data = Wage, family = binomial)
anova(gam2a, gam2, test = "Chisq")


#gam can even plot for lm and glm
