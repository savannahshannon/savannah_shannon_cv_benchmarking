"""
Unit tests for data loading functionality.
Tests: folder, CSV, JSON, and array input formats.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from PIL import Image
from savannah_shannon_cv_benchmarking.data_loader import DataLoader


class TestDataLoader:
    # test data loading from various formats

    @pytest.fixture
    def sample_folder_dataset(self):
        # create a temporary folder structure dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create class folders
            for class_name in ["cat", "dog"]:
                class_dir = tmpdir / class_name
                class_dir.mkdir()
                # Add sample images
                for i in range(5):
                    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
                    img.save(class_dir / f"{class_name}_{i}.jpg")
            yield tmpdir

    @pytest.fixture
    def sample_csv_dataset(self):
        # create a temporary CSV dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # create images
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
                img.save(tmpdir / f"image_{i}.jpg")
            # create CSV
            csv_data = {
                'image_path': [f"image_{i}.jpg" for i in range(10)],
                'label': ['cat' if i < 5 else 'dog' for i in range(10)]
            }
            df = pd.DataFrame(csv_data)
            csv_path = tmpdir / "manifest.csv"
            df.to_csv(csv_path, index=False)
            yield tmpdir

    def test_folder_loader(self, sample_folder_dataset):
        # test loading from folder structure
        loader = DataLoader()
        images, labels, class_names, info = loader.load(
            str(sample_folder_dataset), "folder"
        )

        assert len(images) == 10
        assert len(labels) == 10
        assert len(class_names) == 2
        assert "cat" in class_names
        assert "dog" in class_names

    def test_csv_loader(self, sample_csv_dataset):
        # test loading from CSV manifest
        loader = DataLoader()
        csv_path = sample_csv_dataset / "manifest.csv"
        images, labels, class_names, info = loader.load(
            str(csv_path), "csv"
        )

        assert len(images) == 10
        assert len(labels) == 10
        assert len(class_names) == 2


    @pytest.fixture
    def sample_json_dataset(self):
        # create a temporary JSON dataset with image_path manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # create images
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
                img.save(tmpdir / f"image_{i}.jpg")
            # create JSON manifest with image_path references
            json_data = [
                {
                    'image_path': f"image_{i}.jpg",
                    'label': 'cat' if i < 5 else 'dog'
                }
                for i in range(10)
            ]
            import json
            json_path = tmpdir / "manifest.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            yield tmpdir

    def test_json_loader(self, sample_json_dataset):
        # test loading from JSON manifest with image paths
        loader = DataLoader()
        json_path = sample_json_dataset / "manifest.json"
        images, labels, class_names, info = loader.load(
            str(json_path), "json"
        )

        assert len(images) == 10
        assert len(labels) == 10
        assert len(class_names) == 2
        assert "cat" in class_names
        assert "dog" in class_names

    def test_dataset_validation(self, sample_folder_dataset):
        # test dataset validation checks
        loader = DataLoader()
        images, labels, class_names, info = loader.load(
            str(sample_folder_dataset), "folder"
        )

        # all loaded images should have same shape
        assert all(img.shape == images[0].shape for img in images)
        # Should have at least 2 classes
        assert len(class_names) >= 2

    def test_labels_match_images(self, sample_csv_dataset):
        # test that number of labels equals number of images
        loader = DataLoader()
        csv_path = sample_csv_dataset / "manifest.csv"
        images, labels, class_names, info = loader.load(
            str(csv_path), "csv"
        )

        assert len(images) == len(labels)
