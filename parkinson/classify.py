#it will detect based on certain data of the patient and help the patient to recover without loosing its life
#it is a progressive nervous system disorder theat affects movement leading to shaking , stiffness and difficulty with walking balance and coordination
# it usually begin gradually and get worse over time
#its an nervous base disorder
#affect people mostly above 50

#importing dependencies

##------------------------------------------
import numpy as np                                           #this is useful for making arrays
import pandas as pd                                          #this is useful for creating pandas data frame ,,,, df are structured tables ... it is easily ot analyse then
from sklearn.model_selection import train_test_split         #sklearn is one of the most import when it comes to machine learning   ##  to slpit the data into training and testing train_test_split is used
from sklearn.preprocessing import StandardScaler             #standardscaler is used to pre process the data for this purpose it is used to standardize our data in a common  range
from sklearn import svm                                      # svm means support vector machine we are  going to use this model
from sklearn.metrics import accuracy_score                   #this is used to evaluate our model and give to accuracy of our model

##------------------------------------------

#Data collection and analysis
##loading the data from csv file to panada data frame
park_data=pd.read_csv('parkinsons.csv')  ## LOCATION

##printing the first five rows of df
print(park_data.head())

#WE WILL COVERT NEAGTIVE TOH +VE ALSO

##No of rows and col in data frame
print(park_data.shape)


#getting more info of dataset
#basic thing in dataset
print(park_data.info())

#checking for missing values in each col
print(park_data.isnull().sum())

#### **** if u are having missing values then u can use mean median to fill the data with the missing values

##getting some statical measures about the data



##what is the distrubution of parkinson in data use of status
##distrubution of traget varible it is status
l=park_data['status'].value_counts
#print(l)
#0 means -ve  healthy
#1 means +ve  affected

#grouping data based on traget varible
k=park_data.groupby('status').mean(numeric_only=True)
#print(k)

##data preprocessing
##seprately status from other and removing the id
## seperating feature and target
x=park_data.drop(columns=['name','status'],axis=1) # droping a col we have to give axis =1 and when droping a particular row axis =0

y=park_data['status']

print(x)
print(y)

## spliting the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2) ##we creating frour array   x split into two arry  xtrain rep the train data feature
                                                                                    #xtest represent the testing data feature
                                                                                    ## cressponding label for all the labels will be store in either xtrain or ytrain
                                                                                    #test_size 0.2 means 20% that we want 20% of  data as test and 80% and train its the general amount
                                                                                    ## random state is liek identity number it will split data in random way  random state =2 in that case split will be kinda same
print(x.shape,xtrain.shape,xtest.shape)

print("total data ::trainingdata::testing ")


#data standardizations  we want all the data in same range but it wont  change the mean of data
scaler = StandardScaler()
scaler.fit(xtrain)##it will understand the data

#transforming the data
xtrain=scaler.transform(xtrain) # it will convert all the values in the same range

xtest=scaler.transform(xtest) ## the data with x train along


print(xtrain)
print(xtest)

print("all the value are now in same range")

## training our machine learing model
## we will use svm  support vector machine
 #our model will have 22 dimension

model = svm.SVC(kernel='linear')


 ## svc is support vector classcifyer with classfy the data into classes and another is svr sv regressor is something which gives a paritcular value
 ## training the svm model with training data


model.fit(xtrain,ytrain)

#training part is done now

##model evalutaiton that is accuracy score
##accuracy on training data


xtrain_pred=model.predict(xtrain)
train_data_accuracy=accuracy_score(ytrain,xtrain_pred)
print("accuracy of training data",train_data_accuracy)##model should be above 75


xtest_pred=model.predict(xtest)
test_data_accuracy=accuracy_score(ytest,xtest_pred)
print("accuracy of test data",test_data_accuracy)
#over or under fitting problem has occur   here it is overfitted


#building predictive system
#tuple ()
input= (198.38300,215.20300,193.10400,0.00212,0.00001,0.00113,0.00135,0.00339,0.01263,0.11100,0.00640,0.00825,0.00951,0.01919,0.00119,30.77500,0.465946,0.738703,-7.067931,0.175181,1.512275,0.096320)
##status has removed from this data
print('ho')
##changing input data to numpy array
input_numpy=np.asarray(input)
##converting tuple to numpy array

##reshaping the  array
input_reshape=input_numpy.reshape(1,-1)
# model will excepting 1 value no the 156 values

##standardise the data now
std_data=scaler.transform(input_reshape)

pred=model.predict(std_data)##it will print the status values of this data
print(pred)
if pred[0]==0:
    print("NO")
else:
    print("yes")
print("by harshit")


import pickle
filename='parkmodel.sav'
pickle.dump(model,open(filename,'wb')) #model is svm
loaded=pickle.load(open('parkmodel.sav','rb'))

for column in x.columns:
  print(column)
