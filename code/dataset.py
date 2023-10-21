import argparse
import os
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import greycomatrix, greycoprops

def extract_features(img_path, wavelet_type, level):
    """Extract wavelet-based features from an image."""
    # Load the image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform wavelet transformation
    coeffs = pywt.dwt2(gray, wavelet_type)
    LL, (LH, HL, HH) = coeffs

    # Extract statistical features for each wavelet component
    features = {}
    for component, name in zip([LL, LH, HL, HH], ['LL', 'LH', 'HL', 'HH']):
        features[f'{name}_mean'] = np.mean(component)
        features[f'{name}_min'] = np.amin(component)
        features[f'{name}_max'] = np.amax(component)
        features[f'{name}_std'] = np.std(component)
        features[f'{name}_var'] = np.var(component)
        features[f'{name}_median'] = np.median(component)

        # Energy
        features[f'{name}_energy'] = np.sum(component**2)

        # Entropy
        component_uint8 = (component * 255).astype(np.uint8)
        glcm = greycomatrix(component_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
        features[f'{name}_entropy'] = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

        # GLCM properties
        for prop in ['contrast', 'homogeneity', 'dissimilarity', 'correlation']:
            features[f'{name}_{prop}'] = greycoprops(glcm, prop)[0, 0]

    return features

def generate_dataset(wavelet_type='bior2.4', level=1, input_dir=None, label_mapping=None):
    """
    Generate a dataset with wavelet transformed features from CXR images.

    Parameters:
    - wavelet_type (str): Type of wavelet to use for transformation.
    - level (int): Decomposition level for wavelet transform.
    - input_dir (str): Parent directory containing sub-directories for each disease category.
    - label_mapping (dict, optional): Mapping from directory names to specific labels.

    Returns:
    - df (DataFrame): Combined dataset with features and labels.
    """

    all_features = []

    # Iterate over each sub-directory in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # Check if it's a directory (and not a file)
        if os.path.isdir(subdir_path):
            label = label_mapping.get(subdir, subdir) if label_mapping else subdir

            # Iterate through the CXR images in the sub-directory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jpg'):  # Assuming images are in JPG format
                    img_path = os.path.join(subdir_path, filename)

                    # Extract features using the helper function
                    features = extract_features(img_path, wavelet_type, level)
                    features['label'] = label

                    all_features.append(features)

    # Convert the aggregated features to a DataFrame
    df = pd.DataFrame(all_features)

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset with wavelet transformed features from CXR images.")
    parser.add_argument("--wavelet_type", default="bior2.4", help="Type of wavelet to use for transformation.")
    parser.add_argument("--level", type=int, default=1, help="Decomposition level for wavelet transform.")
    parser.add_argument("--input_dir", required=True, help="Parent directory containing sub-directories for each disease category.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated dataset.")

    args = parser.parse_args()

    df = generate_dataset(wavelet_type=args.wavelet_type, level=args.level, input_dir=args.input_dir)
    output_filename = f"dataset_{args.wavelet_type}_{args.level}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

# example usage: python dataset.py --input_dir /path/to/images --output_dir /path/to/output --wavelet_type bior2.4 --level 1
