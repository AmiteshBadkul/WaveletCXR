import os
import csv
import pandas as pd
import argparse
import ast

def parse_config(config_path):
    """Parse the configuration file and return a dictionary of parameters."""
    with open(config_path, "r") as config_file:
        config_str = config_file.read()
        return ast.literal_eval(config_str)

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from multiple experiment runs.")
    parser.add_argument("--results_root", default="../results/results/", help="Root directory containing the experiment results.")
    parser.add_argument("--output_file", default="../results/final_results.csv", help="Path to save the aggregated results.")
    args = parser.parse_args()

    processed_data = []

    for subfolder in os.listdir(args.results_root):
        subfolder_path = os.path.join(args.results_root, subfolder)

        if os.path.isdir(subfolder_path):
            config_file = None
            for file_name in os.listdir(subfolder_path):
                if file_name.startswith("config_") and file_name.endswith(".txt"):
                    config_file = os.path.join(subfolder_path, file_name)
                    break

            if config_file:
                config = parse_config(config_file)
                wavelet_type = config.get("wavelet_type")
                level = config.get("level")
                algorithm = config.get("algorithm")

                results_filename = f"results_{wavelet_type}_{level}_{algorithm}.csv"
                results_file = os.path.join(subfolder_path, results_filename)

                if os.path.exists(results_file):
                    with open(results_file, "r") as results_csv:
                        csv_reader = csv.DictReader(results_csv)
                        for row in csv_reader:
                            accuracy = float(row["accuracy"])
                            confusion_matrix = ast.literal_eval(row["confusion_matrix"])
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
                print(f"No config file found in folder: {subfolder}")

    df = pd.DataFrame(processed_data)
    df.to_csv(args.output_file, index=False)
    print(f"Aggregated results saved to {args.output_file}")

if __name__ == "__main__":
    main()
