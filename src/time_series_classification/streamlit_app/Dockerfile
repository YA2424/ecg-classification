FROM python:3.11-slim

WORKDIR /app

RUN pip install streamlit pandas numpy plotly

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/time_series_classification/streamlit_app/app.py"]
