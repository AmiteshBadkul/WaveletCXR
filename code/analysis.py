import os
import csv
import pandas as pd
results_root = "../results/results/"

processed_data = []

for subfolder in os.listdir(results_root):
    subfolder_path = os.path.join(results_root, subfolder)

    if os.path.isdir(subfolder_path):

        for file_name in os.listdir(subfolder_path):

            if file_name.startswith("computation_metrics_") and file_name.endswith(".txt"):
                computation_metrics_file = os.path.join(subfolder_path, file_name)

                parts = file_name.split("_")
                if len(parts) == 5:
                    wavelet_type, level, algorithm = parts[2], parts[3], parts[4].split(".")[0]

                    with open(os.path.join(subfolder_path, f"config_{wavelet_type}_{level}_{algorithm}.txt"), "r") as config_file:
                        config_data = config_file.readlines()

                    with open(computation_metrics_file, "r") as metrics_file:
                        computation_metrics_data = metrics_file.read()

                    results_file = os.path.join(subfolder_path, f"results_{wavelet_type}_{level}_{algorithm}.csv")
                    if os.path.exists(results_file):
                        with open(results_file, "r") as results_csv:
                            csv_reader = csv.DictReader(results_csv)
                            for row in csv_reader:

                                accuracy = float(row["accuracy"])
                                confusion_matrix = eval(row["confusion_matrix"])
                                roc_auc = float(row["roc_auc"])
                                pr_auc = float(row["pr_auc"])
                                f1 = float(row["f1"])

                                processed_data.append({
                                    "Folder": subfolder,
                                    "Algorithm": algorithm,
                                    "Wavelet Type": wavelet_type,
                                    "Level": level,
                                    "Accuracy": accuracy,
                                    "Confusion Matrix": confusion_matrix,
                                    "ROC AUC": roc_auc,
                                    "PR AUC": pr_auc,
                                    "F1 Score": f1,
                                })
                else:
                    print(f"File name format is incorrect in folder: {subfolder}")

for data in processed_data:
    print(data)

df = pd.DataFrame(processed_data)
df.to_csv('../results/final_results.csv')
