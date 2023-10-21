# Wavelet Feature Extraction for Thoracic Diseases Detection

## Project Description

This project delves into the potential of wavelet transforms in digital image processing, targeting Chest X-Ray (CXR) scans for thoracic disease detection. Wavelets, mathematical functions that divide signals into time and frequency components, stand as powerful tools in refining image quality and revealing intricate details. Their strength lies in efficiently handling non-stationary signals, making them indispensable for medical imaging, especially in CXR classification. The project originated from the motivation to develop faster diagnostic methods for identifying COVID-19 and evolved to demonstrate a significant reduction in computational costs while maintaining high accuracy levels.

## Features

- Wavelet-based feature extraction from CXR images.
- Dataset generation for different wavelet configurations.
- Training and evaluation of machine learning models, including RandomForest, XGBoost, and Logistic Regression.
- Ablation studies by zeroing out wavelet features.

## Getting Started

### Prerequisites

- Ensure you have [conda](https://docs.conda.io/en/latest/) installed.

### Setting up the Environment

1. Clone the repository:
   ```
   git clone https://github.com/AmiteshBadkul/WaveletCXR
   cd WaveletCXR
   ```

2. Create a conda environment using the provided `environment.yml` file:
   ```
   cd environment
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```
   conda activate waveletCXR
   ```

## Usage

1. To generate a dataset with wavelet transformed features from CXR images:
    ```bash
    # Usage: dataset.py [OPTIONS]

    # Directory containing the CXR images
    --input_dir="/path/to/images"

    # Directory where the generated datasets will be saved
    --output_dir="/path/to/output"

    # Type of wavelet used for dataset generation (default: "bior2.4")
    --wavelet_type="bior2.4"

    # Decomposition level used for dataset generation (default: 1)
    --level=1

    # Example command:
    python dataset.py --input_dir "/path/to/images" --output_dir "/path/to/output" --wavelet_type "bior2.4" --level 1
    ```

2. For automated dataset generation with different wavelet configurations:
   ```
   python dataset_generation.py
   ```

3. To train and evaluate a model based on the provided dataset and algorithm:
    ```bash
    # Usage: main.py [OPTIONS]

    # Type of wavelet used for dataset generation (default: "bior2.4")
    --wavelet_type="bior2.4"

    # Decomposition level used for dataset generation (default: 1)
    --level=1

    # Directory containing the datasets
    --input_dir="/path/to/dataset"

    # Algorithm to use for training (Choices: 'RF', 'XGBoost', 'Logistic')
    --algorithm="RF"

    # Number of trees in the forest (for RF) or Number of boosting rounds (for XGBoost). Default is 100.
    --n_estimators=100

    # Maximum depth of the tree. Default is None.
    --max_depth

    # Learning rate for the algorithm (for XGBoost). Default is 0.1.
    --learning_rate=0.1

    # Maximum number of iterations for convergence (used in Logistic Regression). Default is 100.
    --max_iter=100

    # Directory to save the config, trained model, and results
    --output_dir="/path/to/results"

    # Flag to conduct an ablation study by zeroing out wavelet features (default: "False").
    --ablation="False"

    # Example command:
    python main.py --input_dir "/path/to/dataset" --output_dir "/path/to/results" --algorithm "RF"
    ```
4. For automated training with different wavelet configurations:
    ```bash
    python all_run.py
    ```



### Additional Notes:

- The `trainer.py` script provides functionalities for training and evaluating models based on the wavelet-processed dataset.
- The `utils.py` script contains utility functions. (Note: The content of `utils.py` was not reviewed in detail during this session.)

## Acknowledgments

This project was carried out under the supervision of Dr. Sudha Radhika at [Your Home University]. Special thanks to all collaborators and contributors.
