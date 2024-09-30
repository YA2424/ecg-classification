from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(layout="wide")
st.title("ECG Time Series Classification App")



# Load data
@st.cache_data
def load_data():
    return np.load("data/X_valid.npy"), np.load("data/y_valid.npy")


X_valid, y_valid = load_data()

labels = {0: "Normal", 1: "R-on-T PVC", 2: "PVC", 3: "SP or EB", 4: "UB"}

def predict(data):
    url = "http://localhost:8000/predict"

    # Convert numpy array to list if necessary
    if isinstance(data, np.ndarray):
        data = data.tolist()

    try:
        response = requests.post(url, json={"data": data})
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        return result["predicted_class"], result["confidence"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error making request: {e}")
        return None, None

def display_sample(row, label, predicted_label, confidence):
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(row, title="ECG Signal")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write(f"Actual class: {labels[label]}")
        if predicted_label is not None:
            color = "green" if labels[label] == predicted_label else "red"
            st.write(
                f'Predicted class: <span style="color:{color}">{predicted_label}</span>',
                unsafe_allow_html=True,
            )

            st.write("Confidence scores:")
            sorted_confidence = sorted(zip(labels.values(), confidence), key=lambda x: x[1], reverse=True)
            for class_label, score in sorted_confidence:
                st.write(f"{class_label}: {score:.4f}")
        else:
            st.write("Prediction failed. Please try again.")

if st.button("Generate Random Sample"):
    index = np.random.randint(0, len(X_valid))
    row = X_valid[index, :]
    label = y_valid[index]
    predicted_label, confidence = predict(row)
    display_sample(row, label, predicted_label, confidence)
    st.divider()
