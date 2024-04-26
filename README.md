# ECG Classification with Deep Learning

This repository contains Python code for an example of deep learning model that classifies the ECG5000 dataset. The model is implemented using tensorflow.

# 

## Dataset

The ECG5000 dataset consists of 5000 ECG recordings, each annotated with labels indicating the corresponding cardiac condition or arrhythmia. The dataset is widely used in machine learning research for tasks such as classification and anomaly detection. https://timeseriesclassification.com/description.php?Dataset=ECG5000



## Installation

To set up the project environment, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/amirouyanis/ecg-classification.git
    cd ecg-classification
    ```

2. Create a new virtual environment (recommended):

    ```bash
    python -m venv env
    ```

3. Activate the virtual environment:

    - **Windows:**

    ```bash
    .\env\Scripts\activate
    ```

    - **Linux/macOS:**

    ```bash
    source env/bin/activate
    ```

4. Install Poetry (if not already installed):

    ```bash
    pip install poetry
    ```

5. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

6. (Optional) If you prefer Docker deployment, build the Docker image:

    ```bash
    docker build -t ecg-classification .
    ```

    ```bash
    docker run -p 8501:8501 ecg-classification
    ```
        And go to http://localhost:8501/ 