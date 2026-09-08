"""
Example: Folder Input (Class Subfolders)
Demonstrates benchmark_image_classification with folder structure organization.
Each class is a subfolder containing images.
"""

from savannah_shannon_cv_benchmarking import benchmark_image_classification
from pathlib import Path

# path to dataset organized as class subfolders
dataset_path = Path.home() / "Documents" / "savannah_shannon_cv_benchmarking" / "datasets" / "mnist_folder"

# get class names from folder structure
class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

print("EXAMPLE: Folder Input Format")
print("\nDataset structure:")
print(f"{dataset_path}/")
for class_name in class_names[:3]:
    class_dir = dataset_path / class_name
    num_images = len(list(class_dir.glob("*.png")))
    print(f"  ├── {class_name}/ ({num_images} images)")
print(f"  └── ... ({len(class_names)} classes total)")

print("\nUsage:")
print(f"""
results = benchmark_image_classification(
    dataset="{dataset_path}",
    dataset_type="folder",
    target_labels={class_names},
    color_mode="grayscale"
)
""")

# run benchmark
results = benchmark_image_classification(
    dataset=str(dataset_path),
    dataset_type="folder",
    target_labels=class_names,
    color_mode="grayscale"
)

print("RESULTS")
print(f"\nLoaded {results['dataset_information']['number_of_images']} images from {results['dataset_information']['number_of_classes']} classes")
print(f"Best model: {results['best_model']} (Macro F1: {results['summary'].iloc[0]['Macro F1']:.4f})")
