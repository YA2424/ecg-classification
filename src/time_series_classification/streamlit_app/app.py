import streamlit as st
import pandas as pd
import plotly.express as px
from time_series_classification.modeling.tools import load_best_model
import numpy as np

st.title('ECG Time Series Classification App')

X_valid=np.load("data/X_valid.npy")
y_valid=np.load("data/y_valid.npy")

model=load_best_model()
labels={0:"Normal",
        1:'R-on-T PVC',
        2:'PVC',
        3:'SP or EB',
        4:'UB'}

if st.button('random sample') :
    index=np.random.randint(0, len(X_valid))
    row = X_valid[index]
    label = y_valid[index]
    
    
    # Plot the random time series
    fig = px.line(row)
    st.plotly_chart(fig)
    
    
    
    # Make prediction and display
    y_pred = np.argmax(model.predict(X_valid[index:index+1].astype('float64')),axis=1)
    st.write(f'Predicted class: {labels[y_pred[0]]}')
    st.write(f'Actual class: {labels[label]}')

    