import pandas as pd
import joblib

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# NEW DATA
# ==========================================

new_data = pd.DataFrame({
    "no_of_dependents": [2],
    "income_annum": [5000000],
    "loan_amount": [12000000],
    "loan_term": [10],
    "cibil_score": [750],
    "residential_assets_value": [6000000],
    "commercial_assets_value": [3000000],
    "luxury_assets_value": [10000000],
    "bank_asset_value": [4000000],
    "education_Not Graduate": [0],
    "self_employed_Yes": [0]
})

# ==========================================
# SCALING
# ==========================================

new_data_scaled = scaler.transform(
    new_data
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    new_data_scaled
)

# ==========================================
# RESULT
# ==========================================

if prediction[0] == 1:
    result = "Approved"
else:
    result = "Rejected"

print("\nPrediction Result:")
print(result)