FROM python:3.11-slim


COPY ./ ./
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py"]