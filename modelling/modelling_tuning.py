# ==========================================
# IMPORT LIBRARY
# ==========================================

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "loan_approval_dataset.csv"
)

df.columns = df.columns.str.strip()

df.drop(
    columns=["loan_id"],
    inplace=True
)

# ==========================================
# ENCODING
# ==========================================

df["loan_status"] = (
    df["loan_status"]
    .str.strip()
)

df["loan_status"] = df["loan_status"].map({
    "Approved":1,
    "Rejected":0
})

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
# HYPERPARAMETER TUNING
# ==========================================

param_grid = {
    "n_estimators": [
        100,
        200,
        300
    ],
    "max_depth": [
        10,
        20,
        None
    ],
    "min_samples_split": [
        2,
        5,
        10
    ]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

# ==========================================
# BEST RESULT
# ==========================================

print(
    "Best Parameters:"
)

print(
    grid_search.best_params_
)

print(
    "\nBest Accuracy:"
)

print(
    grid_search.best_score_
)