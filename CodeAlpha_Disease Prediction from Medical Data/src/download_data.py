import pandas as pd
import requests
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

def download_and_save(url, columns, filename, na_values=None):
    print(f"Processing {filename} from {url}...")
    try:
        # Try reading with default utf-8 first, if fails try latin1
        try:
            df = pd.read_csv(url, names=columns, na_values=na_values)
        except UnicodeDecodeError:
            df = pd.read_csv(url, names=columns, na_values=na_values, encoding='ISO-8859-1')
        
        df.to_csv(f'data/{filename}', index=False)
        print(f"Successfully saved data/{filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 1. Heart Disease (Cleveland - Processed)
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
download_and_save(heart_url, heart_columns, 'heart_disease.csv', na_values="?")


# 2. Diabetes (Pima Indians)
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
download_and_save(diabetes_url, diabetes_columns, 'diabetes.csv')

# 3. Breast Cancer (Wisconsin Diagnostic)
breast_cancer_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breast_cancer_columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", 
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", 
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", 
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", 
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]
download_and_save(breast_cancer_url, breast_cancer_columns, 'breast_cancer.csv')

