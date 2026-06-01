import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# MLFLOW TRACKING SERVER
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Loan_Approval_RandomForest"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "loan_approval_dataset.csv"
)

df.columns = df.columns.str.strip()

# ==========================================
# PREPROCESSING
# ==========================================

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
# TRAINING + MLFLOW LOGGING
# ==========================================

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
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

    # ======================================
    # SAVE MODEL
    # ======================================

    joblib.dump(
        model,
        "model.pkl"
    )

    joblib.dump(
        scaler,
        "scaler.pkl"
    )

    # ======================================
    # LOG ARTIFACTS
    # ======================================

    mlflow.log_artifact(
        "model.pkl"
    )

    mlflow.log_artifact(
        "scaler.pkl"
    )

    # ======================================
    # LOG SKLEARN MODEL
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "Model berhasil disimpan ke MLflow"
    )