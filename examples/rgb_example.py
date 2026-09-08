"""
Example: RGB Image Classification
Demonstrates benchmark_image_classification with a three-channel (RGB) dataset.
Uses CIFAR-10 with CSV manifest input.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path

# path to RGB dataset (CIFAR-10 CSV manifest)
csv_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "cifar10_manifest.csv"

print("EXAMPLE: RGB Image Classification")
print(f"Dataset: CIFAR-10 (natural objects)")
print(f"Format: CSV manifest")
print(f"Images: 1500 (150 per class)")
print(f"Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
print()

# run benchmark
results = benchmark_image_classification(
    dataset=str(csv_path),
    dataset_type="csv",
    target_labels="label",
    color_mode="rgb"
)

# display results
print("RESULTS")
print(f"\nDataset loaded: {results['dataset_information']['number_of_images']} images")
print(f"Classes: {results['dataset_information']['number_of_classes']}")
print(f"Image shape: {results['dataset_information']['image_shape']}")
print(f"\nTraining samples: {results['split_information']['training_samples']}")
print(f"Testing samples: {results['split_information']['testing_samples']}")

print("\nBenchmark Summary:")
print(results['summary'].to_string(index=False))

print(f"\nBest Model: {results['best_model']}")
print(f"Accuracy: {results['summary'].iloc[0]['Accuracy']:.4f}")
print(f"Macro F1: {results['summary'].iloc[0]['Macro F1']:.4f}")
