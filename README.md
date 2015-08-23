# Course-Project
Coursera Course Project
library(caret)
library(randomForest)

set.seed(7667)

data <- read.csv("~/Desktop/Data Science/Machine Learning/Course Project/training.csv")
data <- data[sample(1:nrow(data)), ]
data<- data[, colMeans(is.na(data)) < .5 ]
data<- data[, colMeans(data== "") < .5]
data<- data[, -c(1:7)]


intrain <- createDataPartition(data$classe,p=0.7, list=FALSE)
training<- data[intrain, ]
testing <- data[-intrain, ]
intrain<- createDataPartition(training$classe,p=0.7, list=FALSE)
train<- training[intrain, ]
val <- training[-intrain, ] 
train.s <- train[sample(1:9619, size = 500), ]                              

t0<- system.time(modrf <- train(train.s[,-53], train.s[, 53] , method="rf", trControl=trainControl(number=5)))

system.time(
  modrpart <- train(train.s[,-60], train.s[, 60] , method="rpart", trControl=trainControl(number=5))
)

predictions <- predict(modrf, newdata = val)
confusionMatrix(predictions, val$classe)

predictions <- predict(modrpart, newdata = val)
confusionMatrix(predictions, val$classe)


vars <- varImp(modrf, scale = FALSE)
plot(vars, top = 20)


t1<- system.time(modrf1 <- train(train.s[,-53][,vars$importance > 1], train.s[, 53] , method="rf", trControl=trainControl(number=5)))
t2<- system.time(modrf2 <- train(train.s[,-53][,vars$importance > 2], train.s[, 53] , method="rf", trControl=trainControl(number=5)))
t2.5<- system.time(modrf2.5 <- train(train.s[,-53][,vars$importance > 2.5], train.s[, 53] , method="rf", trControl=trainControl(number=5)))
t4<- system.time(modrf4 <- train(train.s[,-53][,vars$importance > 4 ], train.s[, 53] , method="rf", trControl=trainControl(number=5)))
t6<- system.time(modrf6 <- train(train.s[,-56][,vars$importance > 6 ], train.s[, 56] , method="rf", trControl=trainControl(number=5)))
t8 <- system.time(modrf8 <- train(train.s[,-53][,vars$importance > 8 ], train.s[, 53] , method="rf", trControl=trainControl(number=5)))


t1_2<-  system.time(modrf1_2 <- train(train.s[,-60][,vars$importance >1 & vars$importance <2], train.s[, 60] , method="rf", trControl=trainControl(number=5)))

predictions0 <- predict(modrf, newdata = val)
confusionMatrix(predictions0, val$classe)

predictions1 <- predict(modrf1, newdata = val)
confusionMatrix(predictions1, val$classe)

predictions2 <- predict(modrf2, newdata = val)
confusionMatrix(predictions2, val$classe)

predictions2.5 <- predict(modrf2.5, newdata = val)
confusionMatrix(predictions2.5, val$classe)

predictions4 <- predict(modrf4, newdata = val)
confusionMatrix(predictions4, val$classe)

predictions6 <- predict(modrf6, newdata = val)
confusionMatrix(predictions6, val$classe)

predictions8 <- predict(modrf8, newdata = val)
confusionMatrix(predictions8, val$classe)




modF<- train(training[,-53][,vars$importance > 8], training[, 53] , method="rf", trControl=trainControl(number=5))

predictionsF <- predict(modF, newdata = testing)
confusionMatrix(predictionsF, testing$classe)
