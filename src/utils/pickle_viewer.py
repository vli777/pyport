import pickle
file_path = "optuna_cache/anomaly_thresholds.pkl"
with open(file_path, "rb") as f:
    thresholds = pickle.load(f)

print(thresholds)
