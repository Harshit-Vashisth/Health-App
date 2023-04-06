## this one the important alogrithm of supervised learning algorithm
## in supervised learning we feed the data to our madchine learning model and the machine learning model   learning form the  data and it respective label
##labels are most

#we trainn model with several medical info of patients
# and  wheathe the perosn is diabtic or non diabatic

#it then tries to plot the data in a graph once it plot the data  it tries to find the hyperplane which sperate data into two
##via this with predict that the perosn is diabitc or not
#bmi blood gulcose insulim


#importing dependencies
import pandas as pd         ##useful to create df in more structred way
import numpy as np          #make numy array
from sklearn.model_selection import train_test_split   #train test split to split data into training and testing
from sklearn.linear_model import LogisticRegression     #logisctics reg libbary
from sklearn.metrics import accuracy_score      #to evaluate accuracy scroe


## preoprcess the data first
#standardize the  data so that all thsi data lies in the smae range

#split data in train test then same as other
##then feed to svm

data=read_csv('diabetes.csv')
