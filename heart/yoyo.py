import pandas as pd         ##useful to create df in more structred way
import numpy as np          #make numy array
from sklearn.model_selection import train_test_split   #train test split to split data into training and testing
from sklearn.linear_model import LogisticRegression     #logisctics reg libbary
from sklearn.metrics import accuracy_score


data=pd.read_csv("cardio.csv")




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