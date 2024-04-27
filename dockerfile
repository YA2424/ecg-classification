FROM python:3.11-slim


COPY ./ ./
RUN pip install poetry 
RUN poetry install


EXPOSE 8501

CMD ["streamlit", "run", "src/time_series_classification/streamlit_app/app.py"]
