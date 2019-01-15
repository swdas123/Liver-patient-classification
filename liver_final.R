## Read data
data<-read.csv(file="/home/lenovo/Downloads/RSTUDIO_WORKSPACE/ML_NEW/CLASSIFICATION/DROPOUT-PREDICTION/assignment2/indian_liver_patient.csv")

## Explore data frame
str(data)
names(data)
summary(data)  #Albumin_and_Globulin_Ratio has NA values
table(data$Albumin_and_Globulin_Ratio, useNA = "ifany") 
summary(data$Albumin_and_Globulin_Ratio)
class(data$Albumin_and_Globulin_Ratio)

average <- ave(data$Albumin_and_Globulin_Ratio, FUN = function(x) mean(x, na.rm = TRUE))
# Fill all the missing age fields with average
data$Albumin_and_Globulin_Ratio <- ifelse(is.na(data$Albumin_and_Globulin_Ratio), average, data$Albumin_and_Globulin_Ratio)
class(data$Albumin_and_Globulin_Ratio)
table(data$Albumin_and_Globulin_Ratio, useNA = "ifany") 
summary(data$Albumin_and_Globulin_Ratio)

#changing Labels 1,2 to 0,1
table(data$Label)
data$Label=ifelse(data$Label>1,1,0)
table(data$Label)
summary(data)
class(data$Label)
summary(data$Label)

#max-min scaling
data$Total_Bilirubin=(data$Total_Bilirubin-min(data$Total_Bilirubin))/(max(data$Total_Bilirubin)-min(data$Total_Bilirubin))
data$Direct_Bilirubin=(data$Direct_Bilirubin-min(data$Direct_Bilirubin))/(max(data$Direct_Bilirubin)-min(data$Direct_Bilirubin))
data$Alkaline_Phosphotase=(data$Alkaline_Phosphotase-min(data$Alkaline_Phosphotase))/(max(data$Alkaline_Phosphotase)-min(data$Alkaline_Phosphotase))
data$Alamine_Aminotransferase=(data$Alamine_Aminotransferase-min(data$Alamine_Aminotransferase))/(max(data$Alamine_Aminotransferase)-min(data$Alamine_Aminotransferase))
data$Aspartate_Aminotransferase=(data$Aspartate_Aminotransferase-min(data$Aspartate_Aminotransferase))/(max(data$Aspartate_Aminotransferase)-min(data$Aspartate_Aminotransferase))
data$Total_Protiens=(data$Total_Protiens-min(data$Total_Protiens))/(max(data$Total_Protiens)-min(data$Total_Protiens))
data$Albumin=(data$Albumin-min(data$Albumin))/(max(data$Albumin)-min(data$Albumin))
data$Albumin_and_Globulin_Ratio=(data$Albumin_and_Globulin_Ratio-min(data$Albumin_and_Globulin_Ratio))/(max(data$Albumin_and_Globulin_Ratio)-min(data$Albumin_and_Globulin_Ratio))
class(data$Albumin)

data$Gender <- ifelse(data$Gender == "Male" , 0, 1)
summary(data)
#---------------
library(caret)
set.seed(300)
intrain<-createDataPartition(y=data$Label,p=0.8,list = F)
train1<- data[intrain,]
test1<-data[-intrain,]
dim(test1)
dim(train1)
class(test1)
prop.table(table(train1$Label))
prop.table(table(test1$Label))




#---------------------------------------------------------------

# Building Random Forest using R
#---------------------------------------------------------------
set.seed(300)
library(randomForest)

#-------------------------------------
#------using 10 fold cross validatioN
#------------------------------------
#Random Forest that considers full set of features at each split same as a Bagged Decision Tree.
library(caret)
ctrl=trainControl(method = "repeatedcv", number = 10, repeats = 10)
grid_rf=expand.grid(.mtry=c(1,2,4,9))

train1$Label=as.factor(train1$Label) 

#dont run the next 2 line it takes time to execute
# m_rf=train(Label~.,train1,method="rf",metric="Kappa",trControl=ctrl,tuneGrid=grid_rf)
# m_rf
#------------
#outputs
# mtry  Accuracy      Kappa       
# 1     0.7180989824  0.1555141300
# 2     0.7035291397  0.1953906235
# 4     0.6960638298  0.1950507313
# 9     0.6981683626  0.2120551271
#----------------

set.seed(300)
t1 <- randomForest(Label~., train1,method="rf",metric="Kappa",trControl=ctrl,tuneGrid=expand.grid(.mtry=c(9)))

pred_model=predict(t1,test1)
pred_model

# t2 <- randomForest(Label~., train1)  #without 10fold CV
# t2
# predd=predict(t2,test1)
# summary(predd)
# confusionMatrix(table(predd,test1$Label),positive = '1')  #Accuracy : 0.7155172,Kappa : 0.2694656

library(caret)
confusionMatrix(table(pred_model,test1$Label),positive = '1')  #Accuracy : 0.7327586 ,Kappa : 0.3009331   



# #-----
# Applying Gradient Boost
# #---------

class(train1$Label)
summary(train1$Label)
train1$Label=as.numeric(train1$Label)
train1$Label=ifelse(train1$Label>1,1,0)
class(test1$Label)

set.seed(300)

library("gbm")
t3 <- gbm(Label~.,data=train1,distribution = "bernoulli",n.trees = 200,shrinkage = 0.1,interaction.depth = 5)

pred_model1=predict(t3,test1,n.trees = 180)
pred_model1
summary(pred_model1)
pred_model1=ifelse(pred_model1>0.5,1,0)
library(caret)
confusionMatrix(table(pred_model1,test1$Label),positive = '1') #Accuracy : 0.6551724, Kappa : 0.0538336 




#-------------------------------
#----bagging
#--------------------------------
library(ipred)
set.seed(300)
mybag=bagging(Label~.,data=train1,nbagg=25)
pred22=predict(mybag,test1)
pred22
summary(pred22)
pred22=ifelse(pred22>0.5,1,0)
library(caret)
confusionMatrix(table(test1$Label,pred22),mode="everything")  #Accuracy : 0.6810345 ,  Kappa : 0.1497623

#--------bagging with 10 fold CV
library(caret)
set.seed(300)
ctrl=trainControl(method="cv",number=10)
# train1$Label=as.factor(train1$Label) 
# train(Label~.,train1,method="treebag",trControl=ctrl)
# #Accuracy      Kappa       
# #0.7000462535  0.2119299787
mybag1=bagging(Label~.,data=train1,nbagg=25,method="treebag",trControl=ctrl)
pred33=predict(mybag1,test1)
pred33
summary(pred33)
library(caret)
confusionMatrix(table(test1$Label,pred33),mode="everything")

#-----------------------------------------------------
#
#------USING DECISION TREE
#
#-------------------------------------------------------
#train1$Label=as.factor(train1$Label)  #train1$Label converted from numeric to factor
library(C50)
set.seed(300)
model=C5.0(train1[-11],train1$Label)
summary(model)
pred=predict(model,test1)
class(pred)
pred
library(caret)
confusionMatrix(table(test1$Label,pred),mode="everything") #Accuracy : 0.6896552, Kappa : -0.0336634 
#----------------------------
#-------ADAPTIVE BOOSTING using trials
#----------------------------
dim(train1)
set.seed(300)
boost10 <- C5.0(train1[-11], train1$Label,
                trials = 10)
boost10

summary(boost10)

boost_pred10 <- predict(boost10, test1)
class(boost_pred10)
confusionMatrix( table(test1$Label, boost_pred10), mode = "everything")  #Accuracy : 0.6637931, Kappa : 0.1816208



#---------------------------------------------
#
#------------usingSVM
#
#---------------------------------------------
set.seed(300)
library(e1071)
svm_model<- svm(Label~.,data=train1)
summary(svm_model)
pred11<- predict(svm_model,test1)
summary(pred11)
class(pred11)
pred11
library(caret)
confusionMatrix(table(test1$Label,pred11),mode="everything")  #Accuracy : 0.7241379,Kappa : 0.0811881  



#-----------------------------------------------------
#
#------USing NEURAL NETWORK
#
#-----------------------------------------------------

data=as.matrix(data)

ind=sample(2,nrow(data),replace=T,prob = c(0.8,0.2))
training=data[ind==1,]
testing=data[ind==2,]
summary(training)
library(neuralnet)
set.seed(300)

class(training)

n = neuralnet(Label~Age+Gender+Total_Bilirubin+Direct_Bilirubin+Alkaline_Phosphotase+Alamine_Aminotransferase+Aspartate_Aminotransferase+Total_Protiens+Albumin+Albumin_and_Globulin_Ratio,
              data=training,
              hidden=c(13,10,2),
              act.fct = "logistic",
              err.fct = "ce",
              linear.output = FALSE,
              lifesign = "minimal",
              rep=5,
              algorithm = "rprop+")


output= compute(n,testing[,-11])
output

p1=output$net.result
summary(p1)
pred1=ifelse(p1>0.5,1,0)

library(caret)
confusionMatrix(table(pred1,testing[,11]),positive = '1')  # Accuracy : 0.6535433 ,Kappa : 0.2317844    
