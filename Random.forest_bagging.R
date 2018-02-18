##### A Description of the dataset #############
# A data frame with 777 observations on the following 18 variables.

# Private = A factor with levels No and Yes indicating private or public university
# Apps = Number of applications received
# Accept = Number of applications accepted
# Enroll = Number of new students enrolled
# Top10perc = Pct. new students from top 10% of H.S. class
# Top25perc = Pct. new students from top 25% of H.S. class
# F.Undergrad = Number of fulltime undergraduates
# P.Undergrad = Number of parttime undergraduates
# Outstate = Out-of-state tuition
# Room.Board = Room and board costs
# Books = Estimated book costs
# Personal = Estimated personal spending
# PhD = Pct. of faculty with Ph.D.'s
# Terminal = Pct. of faculty with terminal degree
# S.F.Ratio = Student/faculty ratio
# perc.alumni = Pct. alumni who donate
# Expend = Instructional expenditure per student
# Grad.Rate = Graduation rate

# Grad.Rate is our response variable, which we will try to predict using different tree based methods.


######### RANDOM FOREST AND BAGGING IMPLEMENTATION ##############

#### Bagging means that all the predictor variables are used
#install.packages("randomForest")

library(randomForest)
set.seed(26)

data(College)
str(College)

train <- sample(1:nrow(College), 400)

bag.college <- randomForest(Grad.Rate ~., 
                            data=College,
                            subset=train,
                            mtry=17, # no of variables to be considered at each split
                            importance =TRUE)
bag.college

## mtry is set to 17, because this is bagging
## we use all of the variables


## predict the graduation rates using the randomforest model
yhat.bag <- predict(bag.college, newdata = College.test)
plot(yhat.bag, real.data)
abline(0,1)
mean((yhat.bag - real.data)^2)

## MSE = 176.76


## Now lets limit the number of trees using ntree argument
bag2.college <- randomForest(Grad.Rate ~.,
                            data=College,
                            subset=train,
                            mtry=17,
                            ntree=400)
bag2.college

yhat.bag <- predict(bag2.college, newdata=College[-train ,]) # or college.test as before
mean((yhat.bag - real.data)^2)

## MSE = 185.68


####### RANDOM FOREST ################

## Rule for setting number of variables selected from (p)
## in Classification = sqrt(p)
## in Regression = p/3

rf.college <- randomForest(Grad.Rate ~.,
                           data=College,
                           subset=train,
                           importance =TRUE)
rf.college

# the only tuning parameter that is used in a random forest is mtry, which indicates the number 
# of variables that are selected in each split, in each tree. Random Forest algorithm uses a random sample of a subset of 
# variables in each split.

# For example in this dataset or our last random forest model fit, 5 variables are randomly selected
# in each split and the split is confined to one of those variables. This is the way that random forest
# decorrelates trees.

## Now, since we have 17 variables, we will try using mtry from 1 to 17 variables at each split
## and record the error.

oob.err = double(17)
test.err = double(17)

for(mtry in 1:17){
  fit = randomForest(Grad.Rate ~., data=College, subset=train, mtry=mtry, ntree=1000)
  oob.err[mtry] = fit$mse[600]
  pred = predict(fit,College[-train,])
  test.err[mtry] = with(College[-train,], mean((Grad.Rate-pred)^2))
  cat(mtry,"")
} 

# plotting the error rates
matplot(1:mtry, cbind(test.err,oob.err),pch=19,col=c("red","blue"),type="b",
        ylab="Mean Sq Error")

legend("topright",legend=c("Out.of.Bag", "Test"), pch = 19, col = c("red","blue"), text.font = 7)

# the best mtry is somewhere between 4 and 8
rf.college2 <- randomForest(Grad.Rate ~.,
                            data=College[-train,],
                            mtry = 4,
                            importance =TRUE)
rf.college2

