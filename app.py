import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- Load trained model and encoder ----
try:
    rf_model = joblib.load("rf.pkl")
    FEATURE_NAMES = joblib.load("feature_names.pkl")
    OHE_ENCODER = joblib.load("encoder.pkl")
except FileNotFoundError as e:
    st.error(f"Missing required file: {e}")
    st.stop()

st.title("💼 Salary Prediction App")

# ---- Define categorical and numerical columns ----
categorical_cols = ["Gender", "Education Level", "Job Title"]
numerical_cols = ["Age", "Years of Experience"]

# ---- User Inputs ----
user_input = {
    "Age": st.number_input("Age:", min_value=18, max_value=65, value=25),
    "Years of Experience": st.number_input("Years of Experience:", min_value=0, max_value=50, value=1),
    "Gender": st.selectbox("Gender:", ["Male", "Female", "Other"]),
    "Education Level": st.selectbox("Education Level:", ["Bachelor's", "Master's", "PhD"]),
    "Job Title": st.selectbox("Job Title:", ["Software Engineer", "Data Scientist"])
}

# ---- Preprocessing Function ----
def preprocess_inputs(user_input, categorical_cols, numerical_cols):
    df = pd.DataFrame([user_input])

    # Categorical transform
    df_cat = df[categorical_cols]
    cat_encoded = OHE_ENCODER.transform(df_cat)
    encoded_col_names = OHE_ENCODER.get_feature_names_out(categorical_cols)
    df_cat_encoded = pd.DataFrame(cat_encoded, columns=encoded_col_names)

    # Combine
    final_temp = pd.concat([df[numerical_cols], df_cat_encoded], axis=1)

    # Match model feature structure
    final_df = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    for col in final_temp.columns:
        if col in final_df.columns:
            final_df[col] = final_temp[col].values

    return final_df

# ---- Prediction ----
if st.button("Predict Salary"):
    formatted_input = preprocess_inputs(user_input, categorical_cols, numerical_cols)
    prediction = rf_model.predict(formatted_input)
    st.success(f"Predicted Salary: Rs. {prediction[0]:,.2f}")
