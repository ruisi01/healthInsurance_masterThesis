########################################################################################
## Set work directory                                                                 ##
########################################################################################
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

########################################################################################
## Install the packages uncomment this section if packages not installed              ##
########################################################################################
#install.packages("tidyverse")
#install.packages("caret")
#install.packages("glmnet")
#install.packages("gridExtra")
#install.packages("RAMP")
#install.packages("Metrics")
#install.packages("tsensembler")
#install.packages("vtreat")
#install.packages("broom")
#install.packages("MLmetrics")
#install.packages("leaps")
#install.packages("sjstats")
#install.packages("sealasso")

########################################################################################
## Load the packages                                                                  ##
########################################################################################
library(tidyverse)
library(caret)
library(glmnet)
library(gridExtra)
library(RAMP)
library(Metrics)
library(tsensembler)
library(vtreat)
library(broom)
library(MLmetrics)
library(leaps)
library(sjstats)
library(sealasso)

########################################################################################
## Load the data                                                                      ##
########################################################################################
insurance<-read.csv("insurance.csv")
head(insurance)
summary(insurance)
str(insurance)

########################################################################################
## Coding for the categorical variables                                               ##
########################################################################################
insurance$sex    <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)

########################################################################################
## Split the data into training and test set in 80%:20% partition                     ##
########################################################################################
set.seed(562)
training.samples <- insurance$charges %>%
    createDataPartition(p = 0.8, list = FALSE)
train <- insurance[training.samples,  ]
test  <- insurance[-training.samples, ]
dim(train)
dim(test)

########################################################################################
## OLS 10-folds CV                                                                    ##
########################################################################################
fmla  <- charges~age + sex + bmi + children + smoker + region
nRows <- nrow(train)
set.seed(562)
splitPlan <- kWayCrossValidation(nRows, 10, NULL,NULL)
str(splitPlan)

########################################################################################
## Basic linear model                                                                 ##
########################################################################################
model <- lm(fmla, data = train)
test$pred <- predict(model, newdata = test)
RMSE(test$pred, test$charges)

k <- 10 # Number of folds
train$pred.cv <- 0 
models        <- 0

for(i in 1:k) {
  split <- splitPlan[[i]]
  model <- lm(fmla, data = train[split$train,])
  models[i] <- model
  train$pred.cv[split$app] <- predict(model, newdata = train[split$app,])
}

# Predict from test data
test$pred <- predict(lm(fmla, data = test))

# Get the rmse of the test data's predictions
RMSE(test$pred, test$charges)

########################################################################################
## Linear model (OLS)                                                                 ##
########################################################################################
# fit model with training data
head(train)
linear.mod <- lm(charges~age + sex + bmi + children + smoker + region, data=train)
summary(linear.mod)

# predict by linear model using the train and test set
pred.linear_1 <- predict(linear.mod, newdata=train)
pred.linear   <- predict(linear.mod, newdata=test)
# Model performance metrics
data.frame(
    Rsquare = R2(pred.linear_1, train$charges),
    RMSE = RMSE(pred.linear, test$charges)
)

########################################################################################
## Regularized regression methods:lasso, elastic net and ridge regression             ##
########################################################################################
# set input and output for train and test
train.x <- model.matrix(formula(~age + sex + bmi + children + smoker + region), data=train)
test.x  <- model.matrix(formula(~age + sex + bmi + children + smoker + region), data=test )
y       <- train$charges

#################################################################
## Lasso by 10-folds (default) cross-validation                ##
#################################################################
set.seed(562)
(cv.lasso.mod <- cv.glmnet(train.x, y, alpha=1))
plot(cv.lasso.mod)
ggtitle("Lasso Plot")
(best.lambda <- cv.lasso.mod$lambda.min)
coef(cv.lasso.mod)

# Predict by Lasso model using the train and test data
pred.lasso_1 <- predict(cv.lasso.mod, newx=train.x, s=cv.lasso.mod$lambda.min)
pred.lasso   <- predict(cv.lasso.mod, newx=test.x,  s=cv.lasso.mod$lambda.min)
# Goodness-of-fit and model performance metric
data.frame(
    Rsquare = R2(pred.lasso_1, train$charges),
    RMSE = RMSE(pred.lasso, test$charges)
)

#################################################################
## Elastic Net model using 10-folds (default) cross-validation ##
#################################################################
set.seed(562)
(cv.elastic.mod <- cv.glmnet(train.x, y, alpha=0.5))
plot(cv.elastic.mod)
(best.lambda.elastic <- cv.elastic.mod$lambda.min)
coef(cv.elastic.mod)

# predict by Elastic Net model using the train and test data
pred.elastic_1 <- predict(cv.elastic.mod, newx=train.x, s=cv.elastic.mod$lambda.min)
pred.elastic   <- predict(cv.elastic.mod, newx=test.x,  s=cv.elastic.mod$lambda.min)
# Goodness-of-fit and model performance metric
data.frame(
    Rsquare = R2(pred.elastic_1, train$charges),
    RMSE = RMSE(pred.elastic, test$charges)
)

#################################################################
## Ridge regression model by 10-folds cross-validation         ##
#################################################################
set.seed(562)
(cv.ridge.mod <- cv.glmnet(train.x, y, alpha=0))
plot(cv.ridge.mod)
(best.lambda.ridge <- cv.ridge.mod$lambda.min)
coef(cv.ridge.mod)

# Predict using test data
pred.ridge_1 <- predict(cv.ridge.mod, newx=train.x, s=cv.ridge.mod$lambda.min)
pred.ridge   <- predict(cv.ridge.mod, newx=test.x,  s=cv.ridge.mod$lambda.min)

# Goodness-of-fit and model performance metric
data.frame(
    Rsquare = R2(pred.ridge_1, train$charges),
    RMSE = RMSE(pred.ridge, test$charges)
)

########################################################################################
## Results visualization and comparison for linear, lasso, elastic and ridge          ##
########################################################################################
# Plot all predictions (x-axis) against the (charges) for linear models on test data
p1<- ggplot(test, aes(x = pred.linear, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Pred vs charges by linear model (OLS)")
p2<- ggplot(test, aes(x = pred.lasso, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Pred vs charges by Lasso")
p3<- ggplot(test, aes(x = pred.elastic, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Pred vs charges by Elastic Net")
p4<- ggplot(test, aes(x = pred.ridge, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Pred vs charges by Ridge Regression")
(arrange1<-grid.arrange(p1, p2, p3,p4, ncol=2, nrow = 2))

########################################################################################
## As can be seen in the plots of predictions against the real insurance charges, the ##
## points displayed in partitions. Especially for higher charges, the points showed   ##
## binary distributions beside the y=x line, which indicates that there likely to be  ##
## some interactions between variables. Thus, next we do some exploratory analysis by ##
## coloring variables in the plot to check out the significant categorical factors.   ##
########################################################################################
# Coloring levels of categorical predictor

# Check out smoke factor 
# Coloring smoker/non-smoker for test data
color_easy_1 = c("yellow", "green")[test$smoker]
plot(pred.linear, test$charges, col=color_easy_1, xlab="predictions", ylab="charges", type="p")
    title(main="Coloring smoker/non-smoker of test data for linear model")
    legend("topright", inset=.05, title="smoker or not",
         c("non-smoker","smoker"), fill=c("yellow", "green"), horiz=TRUE)
head(test$smoker)

# Check out gender factor
# Coloring men/women for test data
str(test$sex)
color_easy_2 = c("red", "blue")[test$sex]
plot(pred.linear, test$charges, col=color_easy_2, xlab="predictions", ylab="charges", type="p")
title(main="Coloring men/women of test data for linear model")
legend("topright", inset=.05, title="sex",
    c("men","women"), fill=c("blue","red" ), horiz=TRUE)
head(color_easy_2)
head(test$sex)

# Check out region factor
# Coloring regions for test data
str(test$region)
head(test$region)
color_easy_3 = c("yellow", "green", "red", "blue")[test$region]
plot(pred.linear, test$charges, col=color_easy_3, xlab="predictions", ylab="charges", type="p")
title(main="Coloring 4 regions of test data for linear model")
legend("topright", inset=.05, title="regions",
    c("NE","NW","SE","SW"), fill=c("yellow", "green", "red", "blue" ), horiz=TRUE)
head(color_easy_3)
head(test$region)

# Check out children factor
# Coloring children for test data
str(test$children)
max(test$children)
color_easy_4 = c("yellow", "green", "red", "blue","black")[test$children]
plot(pred.linear, test$charges, col=color_easy_4, xlab="predictions", ylab="charges", type="p")
title(main="Coloring different numbers children of test data for linear model")

########################################################################################
## Quadratic model Using Lasso to choose interactions                                 ##
########################################################################################
train.mat <- model.matrix(formula(~ (age + sex + bmi + children + smoker + region)^2 
                                  + I(age^2) + I(bmi^2) + I(children^2)), data=train)
test.mat  <- model.matrix(formula(~ (age + sex + bmi + children + smoker + region)^2 
                                  + I(age^2) + I(bmi^2) + I(children^2)), data=test)
set.seed(562)
(cv.qlasso <- cv.glmnet(train.mat, y, alpha=1))
plot(cv.qlasso)
ggtitle("Quadratic Model Using Lasso")
(best.qlambda <- cv.qlasso$lambda.min)
coef(cv.qlasso)

# Predict using train data and test data 
pred.qlasso_1 <- predict(cv.qlasso, newx=train.mat, s=best.qlambda)
pred.qlasso   <- predict(cv.qlasso, newx=test.mat, s=best.qlambda)
# Goodness-of-fit and model performance metric
data.frame(
  Rsquare = R2(pred.qlasso_1, train$charges),
  RMSE = RMSE(pred.qlasso, test$charges)
)

# Plot the predictions (x-axis) against the outcome (charges) for test data
ggplot(test, aes(x = pred.qlasso, y = charges)) + 
  geom_point() + 
  geom_abline()+
  ggtitle("Predictions against the real charges by quadratic model via Lasso")

########################################################################################
## Quadratic model Using Elastic Net to choose interactions                           ##
########################################################################################
train.mat <- model.matrix(formula(~ (age + sex + bmi + children + smoker + region)^2
                                  + I(age^2) + I(bmi^2) + I(children^2)), data=train)
test.mat  <- model.matrix(formula(~ (age + sex + bmi + children + smoker + region)^2
                                  + I(age^2) + I(bmi^2) + I(children^2)), data=test)
set.seed(562)
(cv.qelastic <- cv.glmnet(train.mat, y, alpha=0.5))
plot(cv.qelastic)
ggtitle("Quadratic Model Using Elastic Net")
(best.qlambda2 <- cv.qelastic$lambda.min)
coef(cv.qelastic)

# Predict by quadratic model via Elastic Net using train and test data
pred.qelastic_1<-predict(cv.qelastic, newx=train.mat, s=best.qlambda2)
pred.qelastic<-predict(cv.qelastic, newx=test.mat, s=best.qlambda2)

# Goodness-of-fit and model performance metric
data.frame(
    Rsquare = R2(pred.qelastic_1, train$charges),
    RMSE = RMSE(pred.qelastic, test$charges)
)
# Plot the predictions (x-axis) against the outcome (charges) for test data
ggplot(test, aes(x = pred.qelastic, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Predictions against the real charges by quadratic model via Elastic Net")

########################################################################################
## RAMP model                                                                         ##
########################################################################################
# Set up input and output
x.train <- cbind(train$age, train$sex, train$bmi, train$children, train$smoker, train$region)
x.test  <- cbind(test$age,  test$sex,  test$bmi,  test$children,  test$smoker,  test$region )
y.train <- train$charges

# Fit RAMP model
RAMP.model <- RAMP(x.train, y.train, family = "gaussian", penalty = "LASSO")
RAMP.model

# predict by RAMP using train and test data 
predict.ramp_1 <- predict(RAMP.model, newdata=x.train)
predict.ramp   <- predict(RAMP.model, newdata=x.test )

# Goodness-of-fit and model performance metric
data.frame(
    Rsquare = R2(predict.ramp_1, train$charges),
    RMSE = RMSE(predict.ramp, test$charges)
)

# Plot the predictions (x-axis) against the outcome (charges) for test data
ggplot(test, aes(x = predict.ramp, y = charges)) + 
    geom_point() + 
    geom_abline()+
    ggtitle("Predictions against the real charges by RAMP")
