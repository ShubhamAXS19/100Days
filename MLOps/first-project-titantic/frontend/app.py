import streamlit as st
import requests

st.title("🚢 Titanic Survival Prediction")

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Collect user input
Pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", {"Male": 0, "Female": 1})
Age = st.selectbox("Age Category", [0, 1, 2, 3, 4])
Fare = st.selectbox("Fare Category", [0, 1, 2, 3])
Embarked = st.selectbox("Embarked", {"S": 0, "C": 1, "Q": 2})
Title = st.selectbox("Title", {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
IsAlone = st.selectbox("Is Alone", {"Yes": 1, "No": 0})

if st.button("Predict"):
    # Prepare payload
    payload = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "Fare": Fare,
        "Embarked": Embarked,
        "Title": Title,
        "IsAlone": IsAlone
    }

    # Send POST request
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        st.success(response.json()["prediction"])
    else:
        st.error("API error: " + str(response.text))
