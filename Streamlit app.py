import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# ---- Load trained Random Forest model ----
model = pickle.load(open("rf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
try:    
     rf_model = joblib.load('rf.pkl')
     FEATURE_NAMES = joblib.load('feature_names.pkl')
     OHE_ENCODER = joblib.load('encoder.pkl')
except FileNotFoundError as e:
    st.error("f 'rf.pkl','feature_names.pkl','encoder.pkl' {e}")
    st.stop()

st.title("💼 Salary Prediction App")

# ---- Define categorical and numerical columns ----
categorical_cols = ["Gender", "Education Level", "Job Title"]
numerical_cols = ["Age", "Years of Experience"]

# ---- User Inputs ----
user_input = {}
user_input["Age"] = st.number_input("Age:", min_value=18, max_value=65, value=25)
user_input["Years of Experience"] = st.number_input("Years of Experience:", min_value=0, max_value=50, value=1)
user_input["Gender"] = st.selectbox("Gender:", ["Male", "Female", "Other"])
user_input["Education Level"] = st.selectbox("Education Level:", ["Bachelor's", "Master's", "PhD"])
user_input["Job Title"] = st.selectbox("Job Title:", ["Software Engineer", "Data Scientist", "HR"])

# ---- Preprocess inputs dynamically ----

def preprocess_inputs(user_input, categorical_cols, numerical_cols):
    
    df = pd.DataFrame([user_input])
    
    df_cat = df[categorical_cols]
    cat_encoded = OHE_ENCODER.transform(df_cat)
    
    encoded_col_names = list(OHE_ENCODER.get_feature_names_out(categorical_cols))
    df_cat_encoded = pd.DataFrame(cat_encoded, columns=encoded_col_names, index=df.index)
    
    final_input_temp = pd.concat([df[numerical_cols], df_cat_encoded], axis=1)
    
    final_input = pd.DataFrame(0.0, index=[0], columns=FEATURE_NAMES) 
    
    for col in final_input_temp.columns:
        if col in final_input.columns:
            final_input[col] = final_input_temp[col].values
    
    return final_input


# ---- Prediction 

if st.button('Predict Salary'):
    
    final_input_data = preprocess_inputs(user_input, categorical_cols, numerical_cols)
    
    predicted_salary = rf_model.predict(final_input_data)
    
    st.success(f"Predicted Salary: Rs.{predicted_salary[0]:,.2f}")

