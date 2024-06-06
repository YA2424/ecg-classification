import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from time_series_classification.modeling.tools import load_best_model

st.title("ECG Time Series Classification App")

X_valid = np.load("data/X_valid.npy")
y_valid = np.load("data/y_valid.npy")

model = load_best_model()
labels = {0: "Normal", 1: "R-on-T PVC", 2: "PVC", 3: "SP or EB", 4: "UB"}

if st.button("random sample"):
    index = np.random.randint(0, len(X_valid))
    row = X_valid[index]
    label = y_valid[index]
    fig = px.line(row)
    st.plotly_chart(fig)

    y_pred = np.argmax(
        model.predict(X_valid[index : index + 1].astype("float32")), axis=1
    )
    actual_label = labels[label]
    predicted_label = labels[y_pred[0]]

    st.write(f"Actual class: {actual_label}")
    if actual_label == predicted_label:
        st.write(
            f'Predicted class:<span style="color:green"> {predicted_label}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            f'Predicted class:<span style="color:red"> {predicted_label}</span>',
            unsafe_allow_html=True,
        )
