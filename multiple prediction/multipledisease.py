# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:50:01 2023

@author: HP
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models
heart=pickle.load(open('E:/git/Health-App/multiple prediction/models/heartmodel.sav','rb')) #rb means reading file as bytes
park=pickle.load(open('E:/git/Health-App/multiple prediction/models/parkmodel.sav','rb'))



#side bar for mavigation    sidebar create
with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',#TITLE OF SIDE BAR
                         
                         
                            ['Heart Disease Prediction',#list for what are diffrent pages we want as we have seen multiple web apps
                             'Parkinsons Disease Prediction'],
                            
                            icons=['activity','person'],
                            
                            default_index=0)#default index =0 means the page which is selected is 0 that is heart


#Heart Disease Prediction page
if(selected=='Heart Disease Prediction'):
#page title
    st.title('Heart Disease Prediction using ML')
    
    #inputing from user   st is for stemlit text_input is for those input fields and the string we want to display about the input level
    age=st.text_input('Age')
    sex=st.text_input('Sex')
    cp=st.text_input('Chest Pain Types')
    trestbps=st.text_input('Resting Blood pressure')
    chol=st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by flourosopy')
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    
    
#parkinson Disease Prediction page
if(selected=='Parkinsons Disease Prediction'):
#page title
    st.title('Parkinsons Disease Prediction using ML')
    