###heart disease prediction

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


