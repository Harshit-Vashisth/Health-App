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
park_model=pickle.load(open('E:/git/Health-App/multiple prediction/models/parkinsons_model.sav','rb'))



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
    
    #getting input data
    #columns for input fields 
    col1,col2,col3=st.columns(3)# as we have 3 coloumns
    
    with col1:
        age=st.text_input('Age')
    with col2:
        sex=st.text_input('Sex')
    with col3:
        cp=st.text_input('Chest Pain Types')    
    with col1:
        trestbps=st.text_input('Resting Blood pressure')
    with col2:
        chol=st.text_input('Serum Cholestoral in mg/dl') 
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')     
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')      
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')         
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
       oldpeak = st.text_input('ST depression induced by exercise')       
    with col2:
       slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
       ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
       thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    #inputing from user   st is for stemlit text_input is for those input fields and the string we want to display about the input level
    
   
  
   
   
    #code for prediction
    # heer we have to create empty string
    heart_dia=''   #we will save the end result
    
    #creating button for prediction
    if st.button('Heart Disease Test Result'):
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        heart_prediction = heart.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
            heart_dia='You are having Heart Disease'
        else:
            heart_dia='You are not having Heart Disease'
    st.success(heart_dia)
   
#parkinson Disease Prediction page
if(selected=='Parkinsons Disease Prediction'):
#page title
    st.title('Parkinsons Disease Prediction using ML')
    

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP-Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP-Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP-Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP-Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP-Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP-RAP')
        
    with col2:
        PPQ = st.text_input('MDVP-PPQ')
        
    with col3:
        DDP = st.text_input('Jitter-DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP-Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP-Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer-APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer-APQ5')
        
    with col3:
        APQ = st.text_input('MDVP-APQ')
        
    with col4:
        DDA = st.text_input('Shimmer-DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    park_dia=''
    
    #creating a button for pred
    if st.button("Results"):
        park_pred = park_model.predict([[fo,fhi,flo,Jitter_percent,Jitter_Abs,RAP,PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if(park_pred[0]==1):
            park_dia='You are having parkinson disease'
        else:
            park_dia='You are not having parkinson disease'
    st.success(park_dia)