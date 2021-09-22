#

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
