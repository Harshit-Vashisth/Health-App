import pandas as pd         ##useful to create df in more structred way
import numpy as np          #make numy array
import sns as sns
from sklearn.model_selection import train_test_split   #train test split to split data into training and testing
from sklearn.linear_model import LogisticRegression     #logisctics reg libbary
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


data=pd.read_csv("cardio.csv")
data['age'] = data['age'].div(365.25).round(2)
print(data)
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
#
# data.drop(data[(data['height'] > data['height'].quantile(0.975)) | (data['height'] < data['height'].quantile(0.025))].index,inplace=True)
# data.drop(data[(data['weight'] > data['weight'].quantile(0.975)) | (data['weight'] < data['weight'].quantile(0.025))].index,inplace=True)
#
# print("Diastilic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))
#
# data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index,inplace=True)
# data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index,inplace=True)
#
# blood_pressure = data.loc[:,['ap_lo','ap_hi']]
# print("Diastilic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))



print(data['cardio'].value_counts())
#1 -.yes(defective)   0 is no

#spliting the feature and the target
x=data.drop(columns=['cardio','id'],axis=1)
y=data['cardio']
scaler=StandardScaler()
x=scaler.fit_transform(x)
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=11)  #20% as test Stratify split the data similar propotion of 0 and 1


#lets check  number of train and test data
print(x.shape,x_train.shape,x_test.shape)

###MODEL TRAINING
##LOGISTIC REGRESION -> also for classification
model =LogisticRegression(penalty='l1', C=0.1, solver='liblinear')  # loading one instance of logistic reg model



##training our machine learning model logistic reg model
model.fit(x_train,y_train)   # it will find the pattern between these feature and the corresponding targets



xtrain_pred=model.predict(x_train)
train_dataacc=accuracy_score(xtrain_pred,y_train)

print("Accuracy on Training data :",train_dataacc)

##accuracy score of testing data
xtest_pred=model.predict(x_test)
test_dataacc=accuracy_score(xtest_pred,y_test)

print("Accuracy on Training data :",test_dataacc)

input_data=(61.8,2,178,95,130,90,3,3,0,0,1)
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
filename='heartmod.sav'
pickle.dump(model,open(filename,'wb'))
loaded=pickle.load(open('heartmod.sav','rb'))

