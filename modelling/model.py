# ==========================================
# IMPORT LIBRARY
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "loan_approval_dataset.csv"
)

df.columns = df.columns.str.strip()

# ==========================================
# DROP ID COLUMN
# ==========================================

df.drop(
    columns=["loan_id"],
    inplace=True
)

# ==========================================
# TARGET ENCODING
# ==========================================

df["loan_status"] = (
    df["loan_status"]
    .str.strip()
)

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# ==========================================
# FEATURE ENCODING
# ==========================================

df["education"] = (
    df["education"]
    .str.strip()
)

df["self_employed"] = (
    df["self_employed"]
    .str.strip()
)

df = pd.get_dummies(
    df,
    columns=[
        "education",
        "self_employed"
    ],
    drop_first=True
)

# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop(
    "loan_status",
    axis=1
)

y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# EVALUATION
# ==========================================

pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

print(
    f"Accuracy: {accuracy:.4f}"
)

# ==========================================
# SAVE ARTIFACTS
# ==========================================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved successfully")