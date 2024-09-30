# ECG Classification with Deep Learning

This repository contains a Python project for ECG (Electrocardiogram) classification using deep learning. The model is implemented using TensorFlow and classifies ECG signals from the ECG5000 dataset. The project includes a REST API for model inference and a Streamlit web application for interactive visualization and classification of ECG signals.

## Features

- Deep learning model for ECG classification
- Hyperparameter tuning using Bayesian optimization with Optuna
- REST API for model inference
- Streamlit web application for interactive ECG classification
- Docker support for easy deployment

## Dataset

The ECG5000 dataset consists of 5000 ECG recordings, each annotated with labels indicating the corresponding cardiac condition or arrhythmia. The dataset is widely used in machine learning research for tasks such as classification and anomaly detection. More information can be found at: [ECG5000 Dataset](https://timeseriesclassification.com/description.php?Dataset=ECG5000)

## Project Structure

- `src/time_series_classification/`
  - `data_processing/`: Scripts for data preprocessing
  - `modeling/`: Model definition, training, and optimization
  - `deployment/`: REST API implementation
  - `streamlit_app/`: Streamlit web application

## Installation

To set up the project environment, follow these steps:

1. **Clone this repository:**

   ```bash
   git clone https://github.com/amirouyanis/ecg-classification.git
   cd ecg-classification
   ```

2. **Create a new virtual environment (recommended):**

   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment:**

   - **Windows:**

   ```bash
   .\env\Scripts\activate
   ```

   - **Linux/macOS:**

   ```bash
   source env/bin/activate
   ```

4. **Install Poetry (if not already installed):**

   ```bash
   pip install poetry
   ```

5. **Install dependencies using Poetry:**

   ```bash
   poetry install
   ```

## Running the Application

There are two ways to run the application:

### 1. Using Docker Compose (Recommended)

This method will start both the REST API and the Streamlit app in separate containers.

1. **Ensure you have Docker and Docker Compose installed on your system.**

2. **Build and run the containers:**

   ```bash
   docker-compose up --build
   ```

3. **Access the Streamlit app by opening a web browser and navigating to:**

   ```plaintext
   http://localhost:8501
   ```

### 2. Running Locally

If you prefer to run the application locally without Docker:

1. **Start the REST API:**

   ```bash
   python src/time_series_classification/deployment/RestAPI.py
   ```

2. **In a new terminal, start the Streamlit app:**

   ```bash
   streamlit run src/time_series_classification/streamlit_app/app.py
   ```

3. **Access the Streamlit app by opening a web browser and navigating to:**

   ```plaintext
   http://localhost:8501
   ```

## Usage

Once the application is running, you can:

- View and classify random ECG samples from the dataset
- Upload your own CSV file containing ECG data for classification
- Adjust the number of samples to display using the sidebar controls

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the contributors and the community for their support.
- Special thanks to the authors of the ECG5000 dataset for providing valuable data for research.
