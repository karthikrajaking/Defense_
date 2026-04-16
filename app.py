import streamlit as st
import numpy as np
import pickle

model, le = pickle.load(open("../models/model.pkl", "rb"))

st.title("SentinelAI - Defence Decision System")

temp = st.slider("Temperature", 0, 100)
movement = st.slider("Movement", 0, 10)
signal = st.slider("Signal Strength", 0, 100)

if st.button("Predict Threat"):
    data = np.array([[temp, movement, signal]])
    pred = model.predict(data)
    result = le.inverse_transform(pred)
    st.success(f"Threat Level: {result[0]}")
