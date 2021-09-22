library(ISLR)
summary(Hitters)

#remove the missing value
Hitters=na.omit(Hitters)
with(Hitters,sum(is.na(Salary)))


##bestsubset reg
#look though all the regs we can use 'leaps'
library(leaps)
regfit.full= regsubsets(Salary~.,data = Hitters)
summary(regfit.full)
#by defulta it is up to 9, here we want 19
regfit.full= regsubsets(Salary~.,data = Hitters, nvmax=19)
reg.summary = summary(regfit.full)
names(reg.summary)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp")
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], pch=20, col="red")

#A plot that can give summary
plot(regfit.full, scale="Cp")
coef(regfit.full,10)







###forward stepwise selection


#we use 'regsets' funtion
library(ISLR)
summary(Hitters)
regfit.fwd = regsubsets(Salary~., data= Hitters, nvmax = 19, method = 'forward')
summary(regfit.fwd)
plot(regfit.fwd, scale = 'Cp')

#use CV method to choose a model
dim(Hitters)
set.seed(1)
train = sample(seq(263),180,replace = FALSE)
train
regfit.fwd = regsubsets(Salary~., data= Hitters[train,], nvmax = 19, method = 'forward')

#MSE for Test error (note we have 19 models)
val.errors=rep(NA, 19)
x.test=model.matrix(Salary~.,data=Hitters[-train,])
for(i in 1:19){
  coefi= coef(regfit.fwd, id =i)
  pred =x.test[,names(coefi)]%*%coefi
  val.errors[i] = mean((Hitters$Salary[-train] - pred)^2)
}
#plot
plot(sqrt(val.errors),ylab="Root MSE", ylim = c(300,400),pch=19,type="b")
points(sqrt(regfit.fwd$rss[-1]/180), col ="blue", pch=19, type="b")
legend("topright",legend = c("Training","Validation"), col=c("blue", "black"), pch=19)


###A funtion for predicting 'regsubsets'
predict.regsubsets=function(object, newdata, id,...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form,newdata)
  coefi = coef(object, id=id)
  mat[, names(coefi)]%*%coefi
  
}



###CV
#10 fold cv
set.seed(11)
folds = sample(rep(1:10, length = nrow(Hitters)))
folds
table(folds)
cv.errors= matrix(NA,10,19)
for( k in 1:10){
  best.fit = regsubsets(Salary~., data = Hitters[folds!=k,], nvmax=19,method ="forward")
  for(i in 1:19){
    pred = predict(best.fit, Hitters[folds == k,],id=i)
    cv.errors[k,i] = mean((Hitters$Salary[folds ==k]-pred)^2)
  }
}
rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")


###Ridge Regression and the Lasso

#Ridge
library(glmnet)
  
