"""
Example: NumPy Array Input
Demonstrates benchmark_image_classification with in-memory NumPy arrays.
Useful for datasets already loaded in memory or from other sources.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
import numpy as np
from sklearn.datasets import load_digits # type: ignore[reportMissingImports]

print("EXAMPLE: NumPy Array Input")

# load sklearn digits dataset as an example
print("\nLoading sklearn digits dataset (handwritten digits 0-9)...")
digits = load_digits()
X = digits.data  # Shape: (1797, 64) - already flattened
y = digits.target  # Shape: (1797,)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# reshape to image format (N, 8, 8, 1) - single channel
X_images = X.reshape(-1, 8, 8, 1).astype(np.uint8) * 16  # scale back to 0-255 range

print(f"Reshaped to image format: {X_images.shape}")

print("\nUsage:")
print(f"""
# X should be shape (N, Height, Width) or (N, Height, Width, Channels)
# y should be class labels (0, 1, 2, ...) or class names

results = benchmark_image_classification(
    dataset=X_images,           # NumPy array of images
    dataset_type="array",
    target_labels=y,            # Array or list of labels
    color_mode="grayscale"      # or "rgb"
)
""")

# run benchmark
results = benchmark_image_classification(
    dataset=X_images,
    dataset_type="array",
    target_labels=y,
    color_mode="grayscale"
)

print("RESULTS")
print(f"\nLoaded {results['dataset_information']['number_of_images']} images")
print(f"Classes: {results['dataset_information']['number_of_classes']}")
print(f"Image shape: {results['dataset_information']['image_shape']}")
print(f"Best model: {results['best_model']} (Macro F1: {results['summary'].iloc[0]['Macro F1']:.4f})")

print("\nBenchmark Summary:")
print(results['summary'].to_string(index=False))
