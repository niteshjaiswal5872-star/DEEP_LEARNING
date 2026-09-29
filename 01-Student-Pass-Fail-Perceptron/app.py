import streamlit as st
import pickle
import pandas as pd
with open("perceptron_model.pkl", "rb") as file:
    data = pickle.load(file)
model = data["model"]
scaler=data["scaler"]
st.title("🎓 Student Pass/Fail Prediction")    
st.write("Prediction using Perceptron")
sh=st.number_input("Study Hours",1,15,5)
att=st.number_input("Attendance(in %)",0,100,50)
if st.button("Predict"):
    input_data = pd.DataFrame([
    {
        "Study_Hours": att,
        "Attendance": sh
    }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("Student is likely to PASS ✅")
    else:
        st.error("Student is likely to FAIL ❌")    
