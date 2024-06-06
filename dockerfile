FROM python:3.11-slim


COPY ./ ./
RUN pip install poetry 
RUN poetry lock
RUN poetry install
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "src/time_series_classification/streamlit_app/app.py"]
