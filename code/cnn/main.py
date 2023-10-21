import os
import argparse
import torch
import pandas as pd
from dataset import get_dataloaders  # Importing from the modified dataset.py
from trainer import initialize_model, train_model, evaluate_model
from utils import measure_computational_load

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a deep learning model on CXR images.")
    parser.add_argument("--data_dir", required=True, help="Root directory containing the datasets.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the trained model and results.")

    args = parser.parse_args()

    # Create dataloaders
    dataloaders = get_dataloaders(args.data_dir, args.batch_size)

    # Initialize model and optimizer
    model = initialize_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer, args.num_epochs)

    # Evaluate the model
    results = evaluate_model(trained_model, dataloaders["val"])
    print(f"Validation Results: {results}")

    # Save the trained model and results
    model_filename = "resnet_model.pth"
    model_path = os.path.join(args.output_dir, model_filename)
    torch.save(trained_model.state_dict(), model_path)

    results_filename = "results.csv"
    results_path = os.path.join(args.output_dir, results_filename)
    pd.DataFrame([results]).to_csv(results_path, index=False)

    config_filename = "config.txt"
    config_path = os.path.join(args.output_dir, config_filename)
    with open(config_path, 'w') as file:
        file.write(str(vars(args)))

if __name__ == "__main__":
    main()
