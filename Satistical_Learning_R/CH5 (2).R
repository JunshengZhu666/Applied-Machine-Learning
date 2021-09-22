library(ISLR)
summary(Default)
attach(Default)

###Q5

#a logistic reg
set.seed(1)
glm.fit = glm(default ~ income + balance, data = Default, family = binomial)

# use CV
FiveB = function() { 
  #i.split
  train = sample(dim(Default)[1], dim(Default)[1]/2)
  #ii.use train data
  glm.fit = glm(default ~ income + balance, data = Default, family = binomial,
                subset = train
                )
  #iii. predict
  glm.pred = rep("No", dim(Default)[1]/2)
  glm.probs = predict(glm.fit, Default[-train, ], type = "response")
  glm.pred[glm.probs > 0.5] = "Yes"
  # iv compute Te
  return(mean(glm.pred != Default[-train, ]$default))
}
FiveB()

#c  
FiveB()
FiveB()
FiveB()

#d add a dummy variable 'student'
#i.split
train = sample(dim(Default)[1], dim(Default)[1]/2)
#ii.use train data
glm.fit = glm(default ~ income + balance + student, data = Default, family = binomial,
              subset = train
)
#iii. predict
glm.pred = rep("No", dim(Default)[1]/2)
glm.probs = predict(glm.fit, Default[-train, ], type = "response")
glm.pred[glm.probs > 0.5] = "Yes"
# iv compute Te
mean(glm.pred != Default[-train, ]$default)
#adding a dummy variable do not affect out Te


###Q6

#a compute se from summary
#b define boot.fn
#c with boot() output se
library(ISLR)
summary(Default)
attach(Default)
#a set.seed(1)
glm.fit = glm(default ~ income + balance, data = Default, family = binomial)
summary(glm.fit)
#b
boot.fn = function(data, index) return(coef(glm(default ~ income + balance, 
                                                data = data, family = binomial, subset = index)))
#c
library(boot)
boot(Default, boot.fn, 50)
#d 
#similar answers to the second and third digits



###Q7
#a logistic using Lag1 and Lag2
#b with the 1st obs of Lag2
#c predict 1st obs from Lag2
#d wirte a loop
#i logistic with LOOCV
#ii probs
#iii pred
#iv determind error
#e average and get Te for LOOCV

library(ISLR)
summary(Weekly)
set.seed(1)
attach(Weekly)
#a 
glm.fit = glm(Direction ~ Lag1 + Lag2, data = Weekly, family = binomial)
summary(glm.fit)

#b
glm.fit = glm(Direction ~ Lag1 + Lag2, data = Weekly[-1, ], family = binomial)
summary(glm.fit)

#c
predict.glm(glm.fit, Weekly[1, ], type = "response") > 0.5
# pred: up true: down

#d
count = rep(0, dim(Weekly)[1])
for (i in 1:(dim(Weekly)[1])){
  glm.fit = glm(Direction ~ Lag1 + Lag2, data = Weekly[-i, ], family = binomial)
  is_up = predict.glm(glm.fit, Weekly[i, ], type = "response") > 0.5
  is_true_up = Weekly[i, ]$Direction == "Up"
  if (is_up != is_true_up)
    count[i] = 1
}
sum(count)
#error = 490

#e
mean(count)
#TE == 45%





###Q8
#a gennerate a data
#b scatterplot X on Y
#c commpute LOOCV error on FOUR model
#d use other seed
#e choice model
#f commnet on models

#a
set.seed(1)
y = rnorm(100)
x = rnorm(100)
y = x - 2*x^2 + rnorm(100)
#n =100, p =2

#b
plot(x, y)

#c
library(boot)
Data = data.frame(x, y)
set.seed(1)
#1
glm.fit = glm(y ~ x)
cv.glm(Data, glm.fit)$delta
#2
glm.fit = glm(y ~ poly(x, 2))
cv.glm(Data, glm.fit)$delta
#3
glm.fit = glm(y ~ poly(x, 3))
cv.glm(Data, glm.fit)$delta
#4
glm.fit = glm(y ~ poly(x, 4))
cv.glm(Data, glm.fit)$delta

#d
#exact the same

#e
#the quadratic polynomial had the lowest LOOCV

#f
summary(glm.fit)
