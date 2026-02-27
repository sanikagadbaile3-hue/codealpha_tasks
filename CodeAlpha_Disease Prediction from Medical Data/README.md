# Disease Prediction from Medical Data

This project implements machine learning models to predict the possibility of three major diseases based on patient data:
- **Heart Disease** (Cleveland dataset)
- **Diabetes** (Pima Indians dataset)
- **Breast Cancer** (Wisconsin Diagnostic dataset)

## Key Features
- **Exploratory Data Analysis**: Preprocessing and cleaning of raw medical datasets.
- **Multiple Algorithms**: Implementation and comparison of SVM, Logistic Regression, Random Forest, and XGBoost.
- **Model Persistence**: Trained models and scalers are saved for future use.
- **GUI Application**: A user-friendly desktop interface for easy disease prediction.

## Project Structure
- `data/`: Raw and processed CSV datasets.
- `models/`: Saved model weights (`.joblib`) and feature scalers.
- `plots/`: Automatically generated EDA and model performance charts.
- `src/`:
  - `download_data.py`: Script to fetch data from UCI Repository.
  - `preprocess.py`: Data cleaning and feature engineering.
  - `train.py`: Model training, evaluation, and selection.
  - `visualize.py`: Script to generate EDA and performance plots.
  - `gui.py`: Premium Tkinter-based desktop application.

- `requirements.txt`: Python dependencies.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Prediction GUI**:
   ```bash
   python src/gui.py
   ```

## Model Performance
- **Heart Disease**: 88.52% Accuracy (Random Forest)
- **Diabetes**: 77.92% Accuracy (Random Forest)
- **Breast Cancer**: 97.37% Accuracy (SVM)
