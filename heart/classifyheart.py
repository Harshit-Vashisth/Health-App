###heart disease prediction
import classifier as classifier
#WORK FLOW

#1 heart data- this data consist of several phealth parameter corresponding to person health
#2 processing the data -- we cannot feed the raw data into out machine learniing model  therfore we will process this data
#3 spliting the data - now we will split the data into training and test data ,,, training data to train or model and test data to test the data
#4 In this case we are going to use logistic regression model becz thisparticular use case in binaary casscification here we are going to classify wheather a person has disease heart or not
#5 in binnary classification model logistic regression model is very useful
#6 now we will do evaluation on our trained model to check its performance sp after we will get a trained loggistic regressin model and to this model when we feed new data our model can predict wheather a person has heart disease or not



##lets understand about our data
# age - age of the patinet
# sex
# chest pain type(4 type)
#cholestoral in mg/dl
#resting bp
#fasting blood sugar ?120
##etc these are based on ecg curves
#angina pain in heart caused due to low blood during excersice
#target 0 ->healthy 1->heart disease

#importing dependencies
import pandas as pd         ##useful to create df in more structred way
import numpy as np          #make numy array
from sklearn.model_selection import train_test_split   #train test split to split data into training and testing
from sklearn.linear_model import LogisticRegression     #logisctics reg libbary
from sklearn.metrics import accuracy_score      #to evaluate accuracy scroe


###DATA COLLECTION AND PROCESSING
#loading the csv to a pandas df

data=pd.read_csv("heart_data.csv")

#Print first 5 rows fo dataset
print(data.head())
print(data.tail())


#number of rows and coloumns in our dataset
print(data.shape)


#getting more info about the data
print(data.info())

##checking for missing values
print(data.isnull().sum())

#alayse the data
#statistical measures about the data
print(data.describe())


#lets see hwo many people have heart disease and how many of them dont have that
#checking the distribution of target varible
print(data['target'].value_counts())
#1 -.yes(defective)   0 is no


#spliting the feature and the target
x=data.drop(columns='target',axis=1)
y=data['target']

print(x)
print(y)

## nwo we have to feed x &  y to our ml model now before doing that but before we have to split our data into training data and test data
#spliting data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=2)  #20% as test Stratify split the data similar propotion of 0 and 1


#lets check  number of train and test data
print(x.shape,x_train.shape,x_test.shape)


###MODEL TRAINING
##LOGISTIC REGRESION -> also for classification
model =LogisticRegression()  # loading one instance of logistic reg model


##training our machine learning model logistic reg model
model.fit(x_train,y_train)   # it will find the pattern between these feature and the corresponding targets

##model evaluation
## ACCURACY SCORE -> model will be asked to predict the target and these prdeict valeu will be compared to the orignal target value

##accuracy on training data
xtrain_pred=model.predict(x_train)
train_dataacc=accuracy_score(xtrain_pred,y_train)

print("Accuracy on Training data :",train_dataacc)

##accuracy score of testing data
xtest_pred=model.predict(x_test)
test_dataacc=accuracy_score(xtest_pred,y_test)

print("Accuracy on Training data :",test_dataacc)

##accuracy score in both shoudl be similar  ,,, overfittting comes there


#BUILDING A PREDICTIVE SYSTEM
input_data=(55,1,0,160,289,0,0,145,1,0.8,1,1,3)
#processing on this data  change input data to numpy array
data_nump=np.asarray(input_data)
data_nump=data_nump.reshape(1,-1)  # otherwise it will assume it to bw 302  as we are predicting for one data point only

pred=model.predict(data_nump)
print(pred)

if(pred[0]==0):
    print("Person does not have heart problem ")
else:
    print("You are having heart problem")


import pickle
filename='heartmodel.sav'
pickle.dump(model,open(filename,'wb'))
loaded=pickle.load(open('heartmodel.sav','rb'))

for column in x.columns:
  print(column)
