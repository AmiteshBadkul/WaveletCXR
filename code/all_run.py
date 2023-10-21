import subprocess
import os

# List of algorithms, wavelet types, and levels to iterate over
algorithms = ['RF', 'XGBoost', 'Logistic']
wavelet_types = ["bior1.1", "bior1.3", "bior2.6", "bior3.1", "bior3.5", "bior6.8"]
levels = [1, 2, 4, 8]

# Input directory
input_dir = '../dataset'

# Output directory
output_dir = '../results'

# Iterate over algorithms, wavelet types, and levels
for algorithm in algorithms:
    for wavelet_type in wavelet_types:
        for level in levels:
            try:
                # Run the main.py script with the current configuration
                subprocess.run(["python", "main.py", "--wavelet_type", wavelet_type, "--level", str(level), "--input_dir", input_dir, "--algorithm", algorithm, "--output_dir", output_dir])
            except Exception as e:
                print(f"Error: {str(e)} - Could not process dataset for algorithm '{algorithm}', wavelet_type '{wavelet_type}', and level '{level}'. Dataset might be missing or unreadable.")
