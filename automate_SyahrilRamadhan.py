import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "loan_approval_dataset.csv"

MODEL_PATH = "model.pkl"

SCALER_PATH = "scaler.pkl"

EXPERIMENT_NAME = "Loan_Approval_Automation"

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

df.columns = df.columns.str.strip()

print(
    f"Dataset Shape: {df.shape}"
)

# ==========================================
# PREPROCESSING
# ==========================================

print("Preprocessing data...")

df.drop(
    columns=["loan_id"],
    inplace=True
)

df["loan_status"] = (
    df["loan_status"]
    .str.strip()
    .map({
        "Approved": 1,
        "Rejected": 0
    })
)

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
# FEATURE TARGET SPLIT
# ==========================================

X = df.drop(
    "loan_status",
    axis=1
)

y = df["loan_status"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("Splitting dataset...")

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

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ==========================================
# MLFLOW CONFIG
# ==========================================

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# ==========================================
# TRAINING
# ==========================================

with mlflow.start_run():

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ======================================
    # PREDICTION
    # ======================================

    y_pred = model.predict(
        X_test_scaled
    )

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    # ======================================
    # LOG PARAMETERS
    # ======================================

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "max_depth",
        20
    )

    mlflow.log_param(
        "min_samples_split",
        2
    )

    # ======================================
    # LOG METRICS
    # ======================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # ======================================
    # SAVE MODEL
    # ======================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # ======================================
    # LOG ARTIFACTS
    # ======================================

    mlflow.log_artifact(
        MODEL_PATH
    )

    mlflow.log_artifact(
        SCALER_PATH
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    # ======================================
    # RESULT
    # ======================================

    print("\n========== RESULT ==========")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        "\nModel saved successfully."
    )

    print(
        "Artifacts logged to MLflow."
    )