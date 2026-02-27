import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_heart_disease():
    print("Preprocessing Heart Disease data...")
    df = pd.read_csv('data/heart_disease.csv')
    
    # Handle missing values (represented as NaN due to na_values="?" in download script)
    # Cleveland dataset has a few missing values in 'ca' and 'thal'
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # Simple binary classification: any value > 0 means disease present
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    df.to_csv('data/heart_disease_processed.csv', index=False)
    print("Saved data/heart_disease_processed.csv")
    return df

def preprocess_diabetes():
    print("Preprocessing Diabetes data...")
    df = pd.read_csv('data/diabetes.csv')
    
    # In this dataset, some columns have 0 where it doesn't make sense (missing values)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    df.to_csv('data/diabetes_processed.csv', index=False)
    print("Saved data/diabetes_processed.csv")
    return df

def preprocess_breast_cancer():
    print("Preprocessing Breast Cancer data...")
    df = pd.read_csv('data/breast_cancer.csv')
    
    # Drop ID column
    df = df.drop('id', axis=1)
    
    # Encode Target (diagnosis): M=1, B=0
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis']) # M=1, B=0 (alphabetical order B then M)
    
    df.to_csv('data/breast_cancer_processed.csv', index=False)
    print("Saved data/breast_cancer_processed.csv")
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    preprocess_heart_disease()
    preprocess_diabetes()
    preprocess_breast_cancer()
