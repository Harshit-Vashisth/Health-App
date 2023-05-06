## this one the important alogrithm of supervised learning algorithm
## in supervised learning we feed the data to our madchine learning model and the machine learning model   learning form the  data and it respective label
##labels are most
import form as form
#we trainn model with several medical info of patients
# and  wheathe the perosn is diabtic or non diabatic

#it then tries to plot the data in a graph once it plot the data  it tries to find the hyperplane which sperate data into two
##via this with predict that the perosn is diabitc or not
#bmi blood gulcose insulim


#importing dependencies
import pandas as pd         ##useful to create df in more structred way
import numpy as np          #make numy array
import sklearn
from sklearn.model_selection import train_test_split   #train test split to split data into training and testing
from sklearn.linear_model import LogisticRegression     #logisctics reg libbary
from sklearn.metrics import accuracy_score      #to evaluate accuracy scroe
from sklearn.preprocessing import StandardScaler
from sklearn import svm

import csv

## preoprcess the data first
#standardize the  data so that all thsi data lies in the smae range

#split data in train test then same as other
##then feed to svm


#data collection and analysis
data=pd.read_csv('../Covid/Covid_data.csv')
print(data)

print(data.head())

##in our data we ahve  pregnancies , glucose level , BloodPressure , SkinThickness(measured from the tricps tells about fat ), serun insulin level ,
# body mass inde(calcutated by diivding weight by height square), diabetes peddigree funciton number which include some kidn of diabetic value , age, outcome

#checking for rows and cloumn int the dataset
print(data.shape)
print("peeople attribute")

#getting staticcal measure of the dataset
print(data.describe())

print(data['COVID-19'].value_counts())
#0 non 1 yes
print("hehe")
print(data.groupby('COVID-19').mean())

#seperating the data
x=data.drop(columns=['Attended Large Gathering','Abroad travel','Visited Public Exposed Places','Family working in Public Exposed Places','Wearing Masks','Sanitization from Market','COVID-19'],axis=1)  # axis 1 for col 0 for row
y=data['COVID-19']

print(x)
print("XXXXX")
# data standardizationn to standardizze the data
scaler=StandardScaler()
scaler.fit(x)

#transforming the data
x=scaler.transform(x)## sacler.fit.tranfomr does it directly


# spliting the data into train and test
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape)
print(xtrain.shape)
print(xtest.shape)
print("sheee")

#training the model
classifer=svm.SVC(kernel='linear')




#training the svm
classifer.fit(xtrain,ytrain)

#model evaluation
xtrainpred= classifer.predict(xtrain)
trainacc= accuracy_score(xtrainpred,ytrain)
print(trainacc)

#on sample data
xtestpred= classifer.predict(xtest)
testacc= accuracy_score(xtestpred,ytest)
print(testacc)
#


input= (0,0,0,0,0,0,0,0,0,0,0,0,0,0)
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

pred=classifer.predict(std_data)##it will print the status values of this data
print(pred)
if pred[0]==0:
    print("NO")
else:
    print("yes")
print("by harshit")



my_dataframe = pd.DataFrame(x)

# Access the columns attribute of the DataFrame

import pickle
filename='covid_model.sav'  #nothing just a file name
pickle.dump(classifer,open(filename,'wb')) #model is svm
loaded=pickle.load(open('covid_model.sav','rb'))

for column in my_dataframe.columns:
    print(column)
