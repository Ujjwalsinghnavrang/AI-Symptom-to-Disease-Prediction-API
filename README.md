# AI Symptom-to-Disease Prediction API

This project is a machine learning model deployed as a Flask API that predicts medical conditions based on symptom descriptions. It uses a scikit-learn classification model trained on a dataset of symptoms and their corresponding diseases (over 1,000 unique conditions).

## Features

* **Machine Learning Model:** A pre-trained `scikit-learn` model (`model.joblib`) that classifies symptom text.
* **Model Evaluation:** The `evaluate_model.py` script runs the model against a test set and generates an accuracy score, classification report, and confusion matrix.
* **REST API:** A lightweight Flask server (`app.py`) that exposes the model through a simple `/predict` endpoint.
* **JSON Interface:** The API accepts symptom text via a JSON POST request and returns the predicted disease.

## Tech Stack

* **Python 3.x**
* **Flask:** For the web API.
* **scikit-learn:** For the machine learning pipeline and model evaluation.
* **Pandas:** For loading and handling the CSV data.
* **Joblib:** For loading the pre-trained model.
* **NumPy:** For numerical operations.
* **Seaborn & Matplotlib:** For data visualization in the evaluation script.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # On Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install flask scikit-learn pandas joblib numpy seaborn matplotlib
    ```

## Usage

There are two main ways to use this project:

### 1. Run the API Server

This will start the web server to accept prediction requests.

```bash
python app.py
