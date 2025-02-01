import pickle

file_name = "epsilon"
with open(f"optuna_cache/{file_name}.pkl", "rb") as f:
    thresholds = pickle.load(f)

print(thresholds)
