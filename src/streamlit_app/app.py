import streamlit as st
import pandas as pd
import plotly.express as px
from src.modeling.model import load_best_model
import numpy as np

st.title('Time Series Classification App')

X_valid=np.load("data/X_valid.npy")
y_valid=np.load("data/y_valid.npy")

model=load_best_model()


if st.button('random sample') :
    index=np.random.randint(0, len(X_valid))
    row = X_valid[index]
    label = y_valid[index]
    
    
    # Plot the random time series
    fig = px.line(row)
    st.plotly_chart(fig)
    
    
    
    # Make prediction and display
    y_pred = model.predict(X_valid[index:index+1].astype('float32'))[0]
    st.write(f'Predicted class: {round(y_pred[0])}, score :{y_pred[0]}')
    st.write(f'Actual class: {label}')

    