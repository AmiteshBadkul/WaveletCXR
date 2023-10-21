import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from trainer import train_evaluate_model, zero_out_wavelet_features
from utils import measure_computational_load
import joblib
import time
import argparse
from sklearn.preprocessing import LabelEncoder

@measure_computational_load
def train_and_evaluate(X_train, y_train, X_test, y_test, algorithm, hyperparams):
    return train_evaluate_model(X_train, y_train, X_test, y_test, algorithm, hyperparams)

def main():
    print('Started Training!')
    parser = argparse.ArgumentParser(description="Train and evaluate a model based on provided dataset and algorithm.")
    parser.add_argument("--wavelet_type", default="bior2.4", help="Type of wavelet used for dataset generation.")
    parser.add_argument("--level", type=int, default=1, help="Decomposition level used for dataset generation.")
    parser.add_argument("--input_dir", required=True, default='../dataset', help="Directory containing the datasets.")
    parser.add_argument("--algorithm", choices=['RF', 'XGBoost', 'Logistic'], required=True, help="Algorithm to use for training.")
    parser.add_argument("--normalize", default=False, help="Normalize the dataset using standard scaling.")

    # Hyperparameters as arguments
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest (RF) or Number of boosting rounds (XGBoost).")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for the algorithm.")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations for convergence (used in Logistic Regression).")

    parser.add_argument("--output_dir", default='../results', required=True, help="Directory to save the config, trained model, and results.")
    parser.add_argument("--ablation", default="False", help="Conduct an ablation study by zeroing out wavelet features.")

    args = parser.parse_args()

    # Create main results folder and experiment-specific subfolder
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_folder = os.path.join(args.output_dir, "results", timestamp)
    os.makedirs(experiment_folder, exist_ok=True)

    # Load the dataset
    dataset_filename = os.path.join(f"{args.wavelet_type}_{args.level}", f"dataset_{args.wavelet_type}_{args.level}.csv")
    dataset_path = os.path.join(args.input_dir, dataset_filename)

    df = pd.read_csv(dataset_path)

    # Check for ablation study
    if args.ablation == 'True':
        df = zero_out_wavelet_features(df.copy())

    X = df.drop("label", axis=1)
    if args.normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    y = df["label"]
    # Convert labels to integer classes
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    (model, results), computation_metrics = train_and_evaluate(X_train, y_train, X_test, y_test, args.algorithm, vars(args))

    # Save the trained model, results, config, and computational load metrics
    model_filename = f"model_{args.wavelet_type}_{args.level}_{args.algorithm}.pkl"
    model_path = os.path.join(experiment_folder, model_filename)
    joblib.dump(model, model_path)

    results_filename = f"results_{args.wavelet_type}_{args.level}_{args.algorithm}.csv"
    results_path = os.path.join(experiment_folder, results_filename)
    pd.DataFrame([results]).to_csv(results_path, index=False)

    config_filename = f"config_{args.wavelet_type}_{args.level}_{args.algorithm}.txt"
    config_path = os.path.join(experiment_folder, config_filename)
    with open(config_path, 'w') as file:
        file.write(str(vars(args)))

    # Save the computational load metrics
    computation_metrics_filename = f"computation_metrics_{args.wavelet_type}_{args.level}_{args.algorithm}.txt"
    computation_metrics_path = os.path.join(experiment_folder, computation_metrics_filename)
    with open(computation_metrics_path, 'w') as file:
        file.write(str(computation_metrics))
    print('Training Completed!')

if __name__ == "__main__":
    main()
