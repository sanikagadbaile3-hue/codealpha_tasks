import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(dataset_path, target_col, disease_name):
    print(f"\n--- Training models for {disease_name} ---")
    df = pd.read_csv(dataset_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/{disease_name}_scaler.joblib')
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    best_acc = 0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"{name}: CV Mean Acc={cv_scores.mean():.4f}, Test Acc={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
    print(f"Selected {best_model_name} as the best model for {disease_name} (Acc: {best_acc:.4f})")
    
    # Save best model
    joblib.dump(best_model, f'models/{disease_name}_model.joblib')
    
    return best_model_name, best_acc

if __name__ == "__main__":
    results = {}
    
    # Heart Disease
    results['Heart Disease'] = train_and_evaluate('data/heart_disease_processed.csv', 'target', 'heart')
    
    # Diabetes
    results['Diabetes'] = train_and_evaluate('data/diabetes_processed.csv', 'Outcome', 'diabetes')
    
    # Breast Cancer
    results['Breast Cancer'] = train_and_evaluate('data/breast_cancer_processed.csv', 'diagnosis', 'breast_cancer')
    
    print("\n\nFinal Report:")
    for disease, (model, acc) in results.items():
        print(f"{disease}: Best Model = {model}, Accuracy = {acc:.4f}")
