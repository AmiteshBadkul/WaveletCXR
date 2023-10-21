import subprocess
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# List of wavelet types and levels to iterate over
#wavelet_types = ["bior1.1", "bior1.3", "bior2.6", "bior3.1", "bior3.5", "bior6.8"]
wavelet_types = ["bior3.1", "bior3.5", "bior6.8"]
levels = [1, 2, 4, 8]

# Input directory
input_dir = "../data"

# Output directory
output_base_dir = "../dataset"

# Iterate over wavelet types and levels
for wavelet_type in wavelet_types:
    for level in levels:
        # Output directory for the current configuration
        output_dir = f"{output_base_dir}/{wavelet_type}_{level}"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run the dataset.py script with the current configuration
        subprocess.run(["python", "dataset.py", "--input_dir", input_dir, "--output_dir", output_dir, "--wavelet_type", wavelet_type, "--level", str(level)])
