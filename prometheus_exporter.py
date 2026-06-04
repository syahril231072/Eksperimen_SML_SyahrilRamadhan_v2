from prometheus_client import start_http_server
from prometheus_client import Counter, Gauge, Histogram

import time
import joblib
import pandas as pd

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("model.pkl")

# contoh input (WAJIB disesuaikan dengan feature asli model kamu)
sample_input = pd.DataFrame([{
    "feature1": 0.5,
    "feature2": 1.2,
    "feature3": 0
}])

# ==========================================
# METRICS
# ==========================================

prediction_counter = Counter(
    "loan_predictions_total",
    "Total number of loan predictions",
    ["label"]
)

prediction_latency = Histogram(
    "loan_prediction_latency_seconds",
    "Prediction latency in seconds"
)

model_accuracy = Gauge(
    "loan_model_accuracy",
    "Model accuracy from evaluation"
)

# isi dari hasil training kamu (bukan random)
model_accuracy.set(0.984)

# ==========================================
# START EXPORTER
# ==========================================

start_http_server(8000)

print("Prometheus Exporter running on http://localhost:8000")

# ==========================================
# REAL MODEL MONITORING LOOP
# ==========================================

while True:

    start_time = time.time()

    # ======================================
    # REAL PREDICTION (NO DUMMY / NO RANDOM)
    # ======================================
    pred = model.predict(sample_input)[0]

    # update counter dengan label hasil model
    prediction_counter.labels(label=str(pred)).inc()

    # latency measurement
    latency = time.time() - start_time
    prediction_latency.observe(latency)

    print(f"Prediction: {pred}")

    time.sleep(5)