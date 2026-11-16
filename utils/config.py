# Hugging Face fallback model, used only if no locally trained model exists
REPO_ID = "Diveshj/thyroid_models"
MODEL_FILENAME = "thyroid_cancer_model.keras"

# Locally trained model + its validation-fitted calibration and thresholds (training/evaluate.py)
LOCAL_MODEL_PATH = "models/fibonacci_net.keras"
LOCAL_MODEL_META = "models/fibonacci_net.json"
BENCHMARK_REPORT = "reports/benchmark.json"
