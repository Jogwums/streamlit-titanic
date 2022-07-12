import streamlit as st
import pandas as pd 
import pickle
import numpy as np 

def load_model():
    with open('saved_random_forest_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data 

data = load_model()

regressor = data["model"]
le_sex = data["le_sex"]
le_class = data["le_class"]
le_embarked = data["le_embarked"]

def prediction():
    st.title("Titanic Survival Prediction")

    st.write("""### We need some info to run predictions""")
    
    sex = ['male', 'female']
    
    coach = ['Third', 'First', 'Second']
    
    embarked = ['S', 'C', 'Q']
    
    
    # User input fields 
    sex = st.selectbox("Gender", sex)

    coach_class = st.selectbox("Class", coach)

    embarked = st.selectbox("Embarked", embarked)

    age = st.slider("Age",0,100,25)
    
    fare = st.slider("Fare",0,150,70)
    
    pclass = st.slider("PClass",1,3,1)


    code = st.button("Calculate Survival")
    
    # 'pclass', 'sex', 'age', 'fare', 'embarked', 'class'
    
    if code:
        X = np.array([[pclass, sex, age, fare, embarked, coach_class]])
        X[:,1] = le_sex.transform(X[:,1])
        X[:,4] = le_embarked.transform(X[:,4])
        X[:,5] = le_class.transform(X[:,5])
        X = X.astype(float)
        
        res = regressor.predict(X)
        
        label = []
        
        st.subheader(F"The passenger survived{res[0]}")
    