"""
Prepare MNIST and CIFAR-10 datasets for benchmarking.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import csv

try:
    from tensorflow import keras
    print("✓ TensorFlow available, downloading datasets...")

    # download MNIST (grayscale, 10 classes)
    print("\n1. Downloading MNIST...")
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mnist_data = np.vstack([x_train, x_test])
    mnist_labels = np.hstack([y_train, y_test])
    print(f"   Loaded {len(mnist_data)} MNIST images")

    # download CIFAR-10 (RGB, 10 classes)
    print("\n2. Downloading CIFAR-10...")
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cifar_data = np.vstack([x_train, x_test])
    # flatten labels properly - they come as (N, 1) from keras
    cifar_labels = np.concatenate([y_train.flatten(), y_test.flatten()])
    print(f"   Loaded {len(cifar_data)} CIFAR-10 images")

except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    exit(1)

base_dir = Path("~/Documents/savannah_shannon_cv_benchmarking").expanduser()
datasets_dir = base_dir / "datasets"
datasets_dir.mkdir(exist_ok=True)

# MNIST: Grayscale, 10 classes
print("PREPARING MNIST (Grayscale)")
mnist_folder = datasets_dir / "mnist_folder"
mnist_folder.mkdir(exist_ok=True)

class_names = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

for class_id, class_name in class_names.items():
    class_dir = mnist_folder / class_name
    class_dir.mkdir(exist_ok=True)

    # get images for this class
    indices = np.where(mnist_labels == class_id)[0]

    # save first 100 images per class (to keep dataset manageable)
    for i, idx in enumerate(indices[:100]):
        img_array = mnist_data[idx]
        # Save as PNG
        from PIL import Image
        img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        img_path = class_dir / f"{class_name}_{i:04d}.png"
        img.save(img_path)

    print(f"{class_name}: {min(100, len(indices))} images")

print(f"\nMNIST folder structure: {mnist_folder}")
print(f"Total images: {sum(len(list((mnist_folder / class_names[i]).glob('*.png'))) for i in range(10))}")

# create CSV manifest for MNIST
mnist_csv = datasets_dir / "mnist_manifest.csv"
csv_rows = []
for class_id, class_name in class_names.items():
    class_dir = mnist_folder / class_name
    for img_path in sorted(class_dir.glob("*.png")):
        # Use relative path from CSV
        rel_path = img_path.relative_to(datasets_dir)
        csv_rows.append({
            'image_path': str(rel_path),
            'label': class_name
        })

pd.DataFrame(csv_rows).to_csv(mnist_csv, index=False)
print(f"MNIST CSV manifest: {mnist_csv}")
print(f"Total entries: {len(csv_rows)}")

# CIFAR-10: RGB, 10 classes
print("PREPARING CIFAR-10 (RGB)")
cifar_folder = datasets_dir / "cifar10_folder"
cifar_folder.mkdir(exist_ok=True)

cifar_class_names = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

for class_id, class_name in cifar_class_names.items():
    class_dir = cifar_folder / class_name
    class_dir.mkdir(exist_ok=True)

    indices = np.where(cifar_labels == class_id)[0]

    for i, idx in enumerate(indices[:150]):
        img_array = cifar_data[idx]  # Shape: (32, 32, 3)
        from PIL import Image
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        img_path = class_dir / f"{class_name}_{i:04d}.png"
        img.save(img_path)

    print(f"{class_name}: {min(150, len(indices))} images")

print(f"\nCIFAR-10 folder structure: {cifar_folder}")
print(f"Total images: {sum(len(list((cifar_folder / cifar_class_names[i]).glob('*.png'))) for i in range(10))}")

# create CSV manifest for CIFAR-10
cifar_csv = datasets_dir / "cifar10_manifest.csv"
csv_rows = []
for class_id, class_name in cifar_class_names.items():
    class_dir = cifar_folder / class_name
    for img_path in sorted(class_dir.glob("*.png")):
        rel_path = img_path.relative_to(datasets_dir)
        csv_rows.append({
            'image_path': str(rel_path),
            'label': class_name
        })

pd.DataFrame(csv_rows).to_csv(cifar_csv, index=False)
print(f"CIFAR-10 CSV manifest: {cifar_csv}")
print(f"Total entries: {len(csv_rows)}")

print("DATASET PREPARATION COMPLETE")
print(f"\nDatasets ready at: {datasets_dir}")
print("\nNext steps:")
print("1. Test MNIST (grayscale): python run_benchmark_mnist.py")
print("2. Test CIFAR-10 (RGB): python run_benchmark_cifar.py")