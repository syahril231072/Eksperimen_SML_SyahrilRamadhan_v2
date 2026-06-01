from prometheus_client import start_http_server
from prometheus_client import Counter, Gauge

import time
import random

# ==========================================
# METRICS
# ==========================================

prediction_counter = Counter(
    "loan_predictions_total",
    "Total number of loan predictions"
)

prediction_latency = Gauge(
    "loan_prediction_latency_seconds",
    "Prediction latency in seconds"
)

# gunakan hasil model terbaik Anda
model_accuracy = Gauge(
    "loan_model_accuracy",
    "Model accuracy from evaluation"
)

# accuracy Random Forest hasil tuning
model_accuracy.set(0.984)

# ==========================================
# START EXPORTER
# ==========================================

start_http_server(8000)

print(
    "Prometheus Exporter running on http://localhost:8000"
)

# ==========================================
# SIMULATION
# ==========================================

while True:

    start_time = time.time()

    # simulasi satu prediksi
    prediction = random.choice(
        ["Approved", "Rejected"]
    )

    prediction_counter.inc()

    latency = (
        time.time()
        - start_time
    )

    prediction_latency.set(
        latency
    )

    print(
        f"Prediction: {prediction}"
    )

    time.sleep(5)