"""
Benchmark MNIST dataset (Grayscale, 10 classes)
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path
import json

datasets_dir = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets"
mnist_folder = datasets_dir / "mnist_folder"
output_dir = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "benchmark_results_mnist"

print("MNIST BENCHMARK - Grayscale, 10 Classes (Folder Input)")

class_names = sorted([d.name for d in mnist_folder.iterdir() if d.is_dir()])

# run benchmark
results = benchmark_image_classification(
    dataset=str(mnist_folder),
    dataset_type="folder",
    target_labels=class_names,
    color_mode="grayscale",
    output_dir=str(output_dir)
)

print("BENCHMARK RESULTS - MNIST")

print("\nDataset Information:")
print(f"Total images: {results['dataset_information']['number_of_images']}")
print(f"Number of classes: {results['dataset_information']['number_of_classes']}")
print(f"Classes: {', '.join(results['dataset_information']['class_names'])}")
print(f"Color mode: {results['dataset_information']['color_mode']}")
print(f"Image shape: {results['dataset_information']['image_shape']}")

print("\nSplit Information:")
print(f"Training samples: {results['split_information']['training_samples']}")
print(f"Testing samples: {results['split_information']['testing_samples']}")
print(f"Random seed: {results['split_information']['random_seed']}")

print("\nBenchmark Summary (Sorted by Macro F1):")
summary_df = results['summary']
print(summary_df.to_string(index=False))

print(f"\nBest Model: {results['best_model']}")
print(f"Macro F1: {summary_df.iloc[0]['Macro F1']:.4f}")
print(f"Training Time: {summary_df.iloc[0]['Training Time (s)']:.4f}s")
print(f"Inference Time: {summary_df.iloc[0]['Inference Time (ms/image)']:.4f}ms/image")

print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
for file in sorted(output_dir.glob("*")):
    if file.is_file():
        print(f"  - {file.name}")

metadata = {
    "dataset": "MNIST",
    "color_mode": "grayscale",
    "number_of_classes": results['dataset_information']['number_of_classes'],
    "total_images": results['dataset_information']['number_of_images'],
    "training_samples": results['split_information']['training_samples'],
    "testing_samples": results['split_information']['testing_samples'],
    "best_model": results['best_model'],
    "best_macro_f1": float(summary_df.iloc[0]['Macro F1']),
    "results_summary": summary_df.to_dict(orient='records')
}

with open(output_dir / "mnist_benchmark_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nMNIST benchmark complete!")