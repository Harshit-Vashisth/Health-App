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


## preoprcess the data first
#standardize the  data so that all thsi data lies in the smae range

#split data in train test then same as other
##then feed to svm


#data collection and analysis
data=pd.read_csv('diabetes.csv')
print(data)

print(data.head())

##in our data we ahve  pregnancies , glucose level , BloodPressure , SkinThickness(measured from the tricps tells about fat ), serun insulin level ,
# body mass inde(calcutated by diivding weight by height square), diabetes peddigree funciton number which include some kidn of diabetic value , age, outcome

#checking for rows and cloumn int the dataset
print(data.shape)
print("peeople attribute")

#getting staticcal measure of the dataset
print(data.describe())

print(data['Outcome'].value_counts())
#0 non 1 yes

print(data.groupby('Outcome').mean())

#seperating the data
x=data.drop(columns='Outcome',axis=1)  # axis 1 for col 0 for row
y=data['Outcome']

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
