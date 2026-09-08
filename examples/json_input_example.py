"""
Example: JSON Manifest Input
Demonstrates benchmark_image_classification with JSON dataset organization.
JSON file contains records with image_path and label fields.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path
import json
import pandas as pd

# create a JSON manifest from the CIFAR-10 CSV
csv_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "cifar10_manifest.csv"
json_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "cifar10_manifest.json"

print("="*80)
print("EXAMPLE 5: JSON Manifest Format")
print("="*80)

# convert CSV to JSON for demonstration
if not json_path.exists():
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)

# show JSON structure
with open(json_path, 'r') as f:
    records = json.load(f)

print("\nJSON Structure (first 3 records):")
print(json.dumps(records[:3], indent=2))
print(f"\nTotal records: {len(records)}")

print("\nUsage:")
print(f"""
results = benchmark_image_classification(
    dataset="{json_path}",
    dataset_type="json",
    target_labels="label",  # Field name containing class labels
    color_mode="rgb"
)
""")

# run benchmark
results = benchmark_image_classification(
    dataset=str(json_path),
    dataset_type="json",
    target_labels="label",
    color_mode="rgb"
)

print("RESULTS")
print(f"\nLoaded {results['dataset_information']['number_of_images']} images from JSON")
print(f"Classes: {', '.join(results['dataset_information']['class_names'])}")
print(f"Best model: {results['best_model']} (Macro F1: {results['summary'].iloc[0]['Macro F1']:.4f})")
