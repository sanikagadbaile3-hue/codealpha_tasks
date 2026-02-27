import joblib
import pandas as pd
import numpy as np

def predict_creditworthiness():
    # Load the assets
    try:
        model = joblib.load('best_credit_scoring_model.pkl')
        scaler = joblib.load('credit_scaler.pkl')
        le_emp = joblib.load('label_encoder_emp.pkl')
        le_dwell = joblib.load('label_encoder_dwell.pkl')
    except FileNotFoundError:
        print("Error: Model assets not found. Please run the notebook first.")
        return

    print("--- Credit Scoring Prediction Tool ---")
    
    # Sample data for demonstration
    # Format: [Age, AnnualIncome, LoanAmount, CreditScore, TotalDebts, ExistingCreditLines, LatePayments, CreditUtilization, DelinquencyFlag, EmploymentStatus, DwellingType, DebtToIncomeRatio]
    
    sample_raw = {
        'Age': 35,
        'AnnualIncome': 75000,
        'LoanAmount': 15000,
        'CreditScore': 720,
        'TotalDebts': 12000,
        'ExistingCreditLines': 4,
        'LatePayments': 0,
        'CreditUtilization': 0.35,
        'DelinquencyFlag': 0,
        'EmploymentStatus': 'Employed',
        'DwellingType': 'Own'
    }
    
    # Formatting features
    df_sample = pd.DataFrame([sample_raw])
    
    # Preprocessing
    df_sample['EmploymentStatus'] = le_emp.transform(df_sample['EmploymentStatus'])
    df_sample['DwellingType'] = le_dwell.transform(df_sample['DwellingType'])
    df_sample['DebtToIncomeRatio'] = df_sample['TotalDebts'] / (df_sample['AnnualIncome'] + 1)
    
    # Scaling
    sample_scaled = scaler.transform(df_sample)
    
    # Prediction
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1]
    
    print(f"\nResults for Sample Applicant:")
    print(f"Income: ${sample_raw['AnnualIncome']}, Credit Score: {sample_raw['CreditScore']}, Utilization: {sample_raw['CreditUtilization']*100}%")
    print(f"Prediction: {'CREDITWORTHY (Low Risk)' if prediction[0] == 1 else 'NOT CREDITWORTHY (High Risk)'}")
    print(f"Probability of being Creditworthy: {probability:.2%}")

if __name__ == "__main__":
    predict_creditworthiness()
