"""
Benchmark CIFAR-10 dataset (RGB, 10 classes)
This demonstrates the package with an RGB dataset using CSV manifest input.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path
import json

datasets_dir = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets"
cifar_csv = datasets_dir / "cifar10_manifest.csv"
output_dir = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "benchmark_results_cifar10"

print("CIFAR-10 BENCHMARK - RGB, 10 Classes (CSV Manifest Input)")

# run benchmark with CSV input
results = benchmark_image_classification(
    dataset=str(cifar_csv),
    dataset_type="csv",
    target_labels="label",
    color_mode="rgb",
    output_dir=str(output_dir)
)

print("BENCHMARK RESULTS - CIFAR-10")

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
    "dataset": "CIFAR-10",
    "color_mode": "rgb",
    "number_of_classes": results['dataset_information']['number_of_classes'],
    "total_images": results['dataset_information']['number_of_images'],
    "training_samples": results['split_information']['training_samples'],
    "testing_samples": results['split_information']['testing_samples'],
    "best_model": results['best_model'],
    "best_macro_f1": float(summary_df.iloc[0]['Macro F1']),
    "results_summary": summary_df.to_dict(orient='records')
}

with open(output_dir / "cifar10_benchmark_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nCIFAR-10 benchmark complete!")