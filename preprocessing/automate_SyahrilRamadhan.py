import pandas as pd

# ==========================================
# PATH
# ==========================================

RAW_DATA_PATH = "namadataset_raw/loan_approval_dataset.csv"

OUTPUT_PATH = (
    "preprocessing/"
    "loan_approval_preprocessed.csv"
)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading raw dataset...")

df = pd.read_csv(
    RAW_DATA_PATH
)

df.columns = df.columns.str.strip()

print(
    f"Original shape: {df.shape}"
)

# ==========================================
# PREPROCESSING
# ==========================================

print("Running preprocessing...")

# Drop ID column
df.drop(
    columns=["loan_id"],
    inplace=True
)

# Target encoding
df["loan_status"] = (
    df["loan_status"]
    .str.strip()
    .map({
        "Approved": 1,
        "Rejected": 0
    })
)

# Clean text columns
df["education"] = (
    df["education"]
    .str.strip()
)

df["self_employed"] = (
    df["self_employed"]
    .str.strip()
)

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=[
        "education",
        "self_employed"
    ],
    drop_first=True
)

print(
    f"Processed shape: {df.shape}"
)

# ==========================================
# SAVE RESULT
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Preprocessed dataset saved to:"
)

print(
    OUTPUT_PATH
)

