"""
Data loading module for various image dataset formats.

Supports folder structure, CSV, JSON, and NumPy array loading.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Union, Dict, Any
from PIL import Image
import json

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as TorchDataLoader, random_split

class DataLoader:

    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    def load(self,
            data_source: Union[str, np.ndarray, pd.DataFrame],
            format_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        # load dataset from various formats
        if format_type == 'auto':
            format_type = self._detect_format(data_source)

        if format_type == 'folder':
            return self._load_folder(data_source)
        elif format_type == 'csv':
            return self._load_csv(data_source)
        elif format_type == 'json':
            return self._load_json(data_source)
        elif format_type in ('array', 'dataframe'):
            return self._load_array(data_source)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _detect_format(self, data_source: Union[str, np.ndarray, pd.DataFrame]) -> str:
        # detect data format automatically
        if isinstance(data_source, np.ndarray):
            return 'array'
        if isinstance(data_source, pd.DataFrame):
            return 'dataframe'
        if isinstance(data_source, str):
            path = Path(data_source)
            if path.suffix.lower() == '.csv':
                return 'csv'
            elif path.suffix.lower() in ('.json', '.jsonl'):
                return 'json'
            elif path.is_dir():
                return 'folder'
        raise ValueError(f"Cannot detect format for: {data_source}")

    def _load_folder(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        images = []
        labels = []
        class_names = []
        skipped_files = 0

        class_folders = sorted([d for d in folder_path.iterdir() if d.is_dir()])

        for class_idx, class_folder in enumerate(class_folders):
            class_names.append(class_folder.name)

            # load images from this class
            for image_path in class_folder.iterdir():
                if image_path.suffix.lower() in self.supported_formats:
                    try:
                        img = Image.open(image_path)
                        images.append(np.array(img))
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Warning: Failed to load {image_path}: {str(e)}")
                        skipped_files += 1

        if len(images) == 0:
            raise ValueError(f"No images found in {folder_path}")

        return (np.array(images), np.array(labels), class_names,
                {'format': 'folder', 'skipped_files': skipped_files, 'n_images': len(images)})

    def _load_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        # load images from CSV (requires 'image_path' and 'label' columns)
        df = pd.read_csv(csv_path)

        if 'image_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'image_path' and 'label' columns")

        images = []
        labels = []
        skipped_files = 0

        for idx, row in df.iterrows():
            try:
                img_path = Path(row['image_path'])
                if not img_path.is_absolute():
                    img_path = Path(csv_path).parent / img_path

                img = Image.open(img_path)
                images.append(np.array(img))
                labels.append(row['label'])
            except Exception as e:
                print(f"Warning: Failed to load {row['image_path']}: {str(e)}")
                skipped_files += 1

        if len(images) == 0:
            raise ValueError("No images loaded from CSV")

        class_names = sorted(list(set(labels)))
        # Map labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        labels = np.array([label_to_idx[label] for label in labels])

        return (np.array(images), labels, class_names,
                {'format': 'csv', 'skipped_files': skipped_files, 'n_images': len(images)})

    def _load_json(self, json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        images = []
        labels = []
        skipped_files = 0

        json_path = Path(json_path)

        # JSONL
        if json_path.suffix == '.jsonl':
            with open(json_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # support both 'image_path' (manifest) and 'image' (embedded array)
                        if 'image_path' in record and 'label' in record:
                            # Load image from file path
                            img_path = Path(record['image_path'])
                            if not img_path.is_absolute():
                                img_path = json_path.parent / img_path
                            img = Image.open(img_path)
                            images.append(np.array(img))
                            labels.append(record['label'])
                        elif 'image' in record and 'label' in record:
                            # load embedded image array
                            img_array = np.array(record['image'], dtype=np.uint8)
                            images.append(img_array)
                            labels.append(record['label'])
                    except Exception as e:
                        print(f"Warning: Failed to parse JSONL record: {str(e)}")
                        skipped_files += 1
        else:
            # JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    records = data.get('records', [])

                for record in records:
                    try:
                        if 'image_path' in record and 'label' in record:
                            img_path = Path(record['image_path'])
                            if not img_path.is_absolute():
                                img_path = json_path.parent / img_path
                            img = Image.open(img_path)
                            images.append(np.array(img))
                            labels.append(record['label'])
                        elif 'image' in record and 'label' in record:
                            img_array = np.array(record['image'], dtype=np.uint8)
                            images.append(img_array)
                            labels.append(record['label'])
                    except Exception as e:
                        print(f"Warning: Failed to parse JSON record: {str(e)}")
                        skipped_files += 1

        if len(images) == 0:
            raise ValueError("No images loaded from JSON")

        class_names = sorted(list(set(labels)))
        # Map labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        labels = np.array([label_to_idx[label] for label in labels])

        return (np.array(images), labels, class_names,
                {'format': 'json', 'skipped_files': skipped_files, 'n_images': len(images)})

    def _load_array(self, array_data: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        # load from NumPy array or Pandas DataFrame
        if isinstance(array_data, pd.DataFrame):
            # Assume first column is images, second is labels
            images = array_data.iloc[:, 0].values
            labels = array_data.iloc[:, 1].values
        else:
            # assume last axis is labels or separate
            images = array_data
            labels = None

        if labels is None:
            raise ValueError("Labels not found in array data")

        class_names = sorted(list(set(labels)))
        # map labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        labels = np.array([label_to_idx[label] for label in labels])

        return (np.array(images), labels, class_names,
                {'format': 'array', 'n_images': len(images), 'skipped_files': 0})


# PyTorch DataLoader utilities for CNN benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_data_loaders(dataset_name: str = 'cifar10',
                     batch_size: int = 128,
                     data_root: str = './datasets',
                     num_workers: int = 4,
                     seed: int = 42) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
    """
    Load dataset and return train, validation, and test DataLoaders.
    
    Args:
        dataset_name: Name of dataset ('cifar10' or 'mnist')
        batch_size: Batch size for training
        data_root: Root directory for datasets
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    torch.manual_seed(seed)
    
    # Define transforms for different datasets
    if dataset_name.lower() == 'cifar10':
        # CIFAR-10: 32x32 images, need to resize to 224x224 for most pretrained models
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, 
                                                     download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                                    download=True, transform=test_transform)
        
    elif dataset_name.lower() == 'mnist':
        # MNIST: 28x28 grayscale, resize to 224x224 and convert to RGB
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load MNIST
        train_dataset = torchvision.datasets.MNIST(root=data_root, train=True,
                                                   download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(root=data_root, train=False,
                                                  download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split training data into train (80%) and validation (20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_data, val_data = random_split(train_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))
    
    # Create DataLoaders
    train_loader = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = TorchDataLoader(val_data, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
