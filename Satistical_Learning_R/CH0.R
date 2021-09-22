###vectors, data, matrices, subsetting
x=c(2,7,5)
x
y=seq(from=4,length=3,by=3)
?seq
x[2:3]
x[-2]
z=matrix(seq(1,12),4,3)
z
z[,1]
dim(z)
ls()
x=runif(50)
y=rnorm(50)
plot(x,y)
plot(x, y, xlab="random uniform", ylab="random normal",pch="*", col="blue")
par(mfrow=c(2,1))
plot(x,y)
hist(y)
par(mfrow=c(1,1))
### reading in data
auto=read