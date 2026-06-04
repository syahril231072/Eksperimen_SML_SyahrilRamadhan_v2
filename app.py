from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict():

    sample = pd.DataFrame({
        "no_of_dependents":[2],
        "income_annum":[5000000],
        "loan_amount":[12000000],
        "loan_term":[10],
        "cibil_score":[750],
        "residential_assets_value":[6000000],
        "commercial_assets_value":[3000000],
        "luxury_assets_value":[10000000],
        "bank_asset_value":[4000000],
        "education_Not Graduate":[0],
        "self_employed_Yes":[0]
    })

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    return {
        "prediction": int(prediction[0])
    }