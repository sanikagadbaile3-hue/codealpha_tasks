import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Ensure directory for plots
os.makedirs('plots', exist_ok=True)

def generate_eda_plots(df_path, name):
    print(f"Generating EDA plots for {name}...")
    df = pd.read_csv(df_path)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Feature Correlation - {name}')
    plt.tight_layout()
    plt.savefig(f'plots/{name}_correlation.png')
    plt.close()
    
    # 2. Target Distribution
    target_col = 'target' if 'target' in df.columns else ('Outcome' if 'Outcome' in df.columns else 'diagnosis')
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Target Distribution - {name}')
    plt.savefig(f'plots/{name}_distribution.png')
    plt.close()

def generate_performance_plots(disease_name, data_path, target_col):
    print(f"Generating performance plots for {disease_name}...")
    try:
        model = joblib.load(f'models/{disease_name}_model.joblib')
        scaler = joblib.load(f'models/{disease_name}_scaler.joblib')
        df = pd.read_csv(data_path)
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {disease_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'plots/{disease_name}_confusion_matrix.png')
        plt.close()
        
        # 2. ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {disease_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'plots/{disease_name}_roc_curve.png')
            plt.close()
            
    except Exception as e:
        print(f"Error generating performance plots for {disease_name}: {e}")

if __name__ == "__main__":
    # EDA Plots
    generate_eda_plots('data/heart_disease_processed.csv', 'heart')
    generate_eda_plots('data/diabetes_processed.csv', 'diabetes')
    generate_eda_plots('data/breast_cancer_processed.csv', 'breast_cancer')
    
    # Performance Plots
    generate_performance_plots('heart', 'data/heart_disease_processed.csv', 'target')
    generate_performance_plots('diabetes', 'data/diabetes_processed.csv', 'Outcome')
    generate_performance_plots('breast_cancer', 'data/breast_cancer_processed.csv', 'diagnosis')
