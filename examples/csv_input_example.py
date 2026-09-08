"""
Example: CSV Manifest Input
Demonstrates benchmark_image_classification with CSV dataset organization.
CSV contains columns: image_path and label (or custom label column name).
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path
import pandas as pd

# path to CSV manifest
csv_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "cifar10_manifest.csv"

# show CSV structure
print("EXAMPLE: CSV Manifest Format")

df = pd.read_csv(csv_path)
print("\nCSV Structure (first 5 rows):")
print(df.head().to_string(index=False))
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {', '.join(df.columns)}")

print("\nUsage:")
print(f"""
results = benchmark_image_classification(
    dataset="{csv_path}",
    dataset_type="csv",
    target_labels="label",  # Column name containing class labels
    color_mode="rgb"
)
""")

# run benchmark
results = benchmark_image_classification(
    dataset=str(csv_path),
    dataset_type="csv",
    target_labels="label",
    color_mode="rgb"
)

print("RESULTS")
print(f"\nLoaded {results['dataset_information']['number_of_images']} images from CSV")
print(f"Classes: {', '.join(results['dataset_information']['class_names'])}")
print(f"Best model: {results['best_model']} (Macro F1: {results['summary'].iloc[0]['Macro F1']:.4f})")
