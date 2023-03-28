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
    selected=option_menu('Multiple Disease Prediction System',
                            ['Heart Disease Prediction',#list for what are diffrent pages we want as we have seen multiple web apps
                             'Parkinsons Disease Prediction'],
                            icons=['activity','P circle'],
                            default_index=0)#default index =0 means the page which is selected is 0 that is heart


#Heart Disease Prediction page
if(selected=='Heart Disease Prediction'):
#page title
    st.title('Heart Disease Prediction using ML')
    
    
#parkinson Disease Prediction page
if(selected=='Parkinsons Disease Prediction'):
#page title
    st.title('Parkinsons Disease Prediction using ML')
    