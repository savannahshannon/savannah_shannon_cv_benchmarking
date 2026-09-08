"""
Example: Grayscale Image Classification
Demonstrates benchmark_image_classification with a single-channel (grayscale) dataset.
Uses MNIST with folder structure input.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path

# path to grayscale dataset (MNIST folder structure)
dataset_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "mnist_folder"

# get class names from folder structure
class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

print("EXAMPLE: Grayscale Image Classification")
print(f"Dataset: MNIST (handwritten digits)")
print(f"Format: Folder structure")
print(f"Images: 1000 (100 per class)")
print(f"Classes: {', '.join(class_names)}")
print()

# run benchmark
results = benchmark_image_classification(
    dataset=str(dataset_path),
    dataset_type="folder",
    target_labels=class_names,
    color_mode="grayscale"
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
