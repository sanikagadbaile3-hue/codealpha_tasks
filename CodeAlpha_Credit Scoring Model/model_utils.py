import pandas as pd
import numpy as np

def generate_credit_data(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # Numerical features
    age = np.random.randint(21, 70, n_samples)
    annual_income = np.random.randint(20000, 150000, n_samples)
    loan_amount = np.random.randint(1000, 50000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    total_debts = np.random.randint(0, 100000, n_samples)
    existing_credit_lines = np.random.randint(1, 15, n_samples)
    late_payments = np.random.randint(0, 10, n_samples)
    credit_utilization = np.random.uniform(0.1, 1.2, n_samples) # 10% to 120%
    delinquency_flag = (late_payments > 2).astype(int)
    
    # Categorical features
    employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Student'], n_samples, p=[0.7, 0.15, 0.1, 0.05])
    dwelling_type = np.random.choice(['Own', 'Rent', 'Boarding'], n_samples, p=[0.5, 0.4, 0.1])
    
    # Creating a DataFrame
    df = pd.DataFrame({
        'Age': age,
        'AnnualIncome': annual_income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'TotalDebts': total_debts,
        'ExistingCreditLines': existing_credit_lines,
        'LatePayments': late_payments,
        'CreditUtilization': credit_utilization,
        'DelinquencyFlag': delinquency_flag,
        'EmploymentStatus': employment_status,
        'DwellingType': dwelling_type
    })
    
    # Target variable (Creditworthy: 1 for Good, 0 for Bad)
    # Logic: High income, high credit score, low utilization, low late payments -> Good
    z = (0.2 * (annual_income / 150000) + 
         0.3 * (credit_score / 850) - 
         0.2 * (credit_utilization / 1.0) - 
         0.3 * (late_payments / 10) - 
         0.1 * (total_debts / (annual_income + 1)) -
         0.1 * (loan_amount / 50000) +
         np.random.normal(0, 0.05, n_samples))
    
    df['Target'] = (z > 0.1).astype(int)
    
    # Add some missing values for preprocessing practice
    mask = np.random.rand(n_samples) < 0.05
    df.loc[mask, 'AnnualIncome'] = np.nan
    
    return df

if __name__ == "__main__":
    df = generate_credit_data()
    df.to_csv('simulated_credit_data.csv', index=False)
    print("Simulated dataset created: simulated_credit_data.csv")
