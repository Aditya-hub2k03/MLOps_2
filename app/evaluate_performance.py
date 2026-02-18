import pandas as pd

df = pd.read_csv("performance_logs.csv")

accuracy = (df["true_label"] == df["predicted_label"]).mean()

print("Post-Deployment Accuracy:", accuracy)
