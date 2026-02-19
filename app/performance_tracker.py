import csv
import os

FILE = "performance_logs.csv"

if not os.path.exists(FILE):
    with open(FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label"])

def log_performance(true_label, predicted_label):
    with open(FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([true_label, predicted_label])
