import streamlit as st
import pandas as pd 
import pickle
import numpy as np 

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def load_model():
    with open('saved_random_forest_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data 

data = load_model()

regressor = data["model"]
le_sex = data["le_sex"]
le_embarked = data["le_embarked"]

def prediction():
    st.title("Titanic Survival Prediction")
    
    sex = ['male', 'female']
    
    embarked = ['S', 'C', 'Q']
    
    
    # User input fields 
    side = st.sidebar

    side.write("""### We need some info to run predictions""")
    
    sex = side.selectbox("Gender", sex)

    embarked = side.selectbox("Embarked", embarked)

    age = side.slider("Age",0,100,25)
    
    fare = side.slider("Fare",0,150,70)
    
    pclass = side.slider("PClass",1,3,1)


    code = side.button("Calculate Survival")
    
    # 'pclass', 'sex', 'age', 'fare', 'embarked', 'class'
    
    if code:
        X = np.array([[pclass, sex, age, fare, embarked]])
        X[:,1] = le_sex.transform(X[:,1])
        X[:,4] = le_embarked.transform(X[:,4])
        X = X.astype(float)
        
        res = regressor.predict(X)
        
        label = []
        if res == 1:
            label.append('Survived')
        else:
            label.append('did not survive') 
        
        # st.subheader(F"The passenger survived {res[0]}")
        st.subheader(f"The Passenger {label[0]}")

prediction()