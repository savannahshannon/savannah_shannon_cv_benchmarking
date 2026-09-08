"""
Unit tests for image preprocessing functionality.
Tests: resizing, normalization, channel conversion, etc.
"""

import pytest
import numpy as np
from savannah_shannon_cv_benchmarking.preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    # test image preprocessing operations

    @pytest.fixture
    def preprocessor_rgb(self):
        return ImagePreprocessor(image_size=64, color_mode="rgb")

    @pytest.fixture
    def preprocessor_grayscale(self):
        return ImagePreprocessor(image_size=64, color_mode="grayscale")

    @pytest.fixture
    def sample_images_rgb(self):
        # create sample RGB images (N, H, W, 3)
        return np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_images_grayscale(self):
        # create sample grayscale images (N, H, W, 1)
        return np.random.randint(0, 256, (10, 32, 32, 1), dtype=np.uint8)

    def test_image_resizing(self, preprocessor_rgb, sample_images_rgb):
        # test images are resized to correct dimensions
        processed = preprocessor_rgb.preprocess_batch(sample_images_rgb)

        assert processed.shape[1] == 64  # height
        assert processed.shape[2] == 64  # width
        assert len(processed) == 10       # same number of images

    def test_rgb_output_shape(self, preprocessor_rgb, sample_images_rgb):
        # test RGB output has 3 channels
        processed = preprocessor_rgb.preprocess_batch(sample_images_rgb)

        assert processed.shape[3] == 3  # 3 channels

    def test_grayscale_output_shape(self, preprocessor_grayscale, sample_images_grayscale):
        # test grayscale output has 1 channel
        processed = preprocessor_grayscale.preprocess_batch(sample_images_grayscale)

        assert processed.shape[3] == 1  # 1 channel

    def test_normalization_range(self, preprocessor_rgb, sample_images_rgb):
        # test pixel values are normalized to [0, 1]
        processed = preprocessor_rgb.preprocess_batch(sample_images_rgb)

        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_consistent_preprocessing(self, preprocessor_rgb, sample_images_rgb):
        # test preprocessing is deterministic
        result1 = preprocessor_rgb.preprocess_batch(sample_images_rgb.copy())
        result2 = preprocessor_rgb.preprocess_batch(sample_images_rgb.copy())

        np.testing.assert_array_equal(result1, result2)

    def test_flattened_features_shape(self, preprocessor_rgb, sample_images_rgb):
        # test flattened features have correct shape
        processed = preprocessor_rgb.preprocess_batch(sample_images_rgb)
        flattened = preprocessor_rgb.create_flattened_features(processed)

        # RGB: 64 * 64 * 3 = 12,288
        assert flattened.shape == (10, 12288)

    def test_flattened_features_grayscale(self, preprocessor_grayscale, sample_images_grayscale):
        # test flattened features for grayscale
        processed = preprocessor_grayscale.preprocess_batch(sample_images_grayscale)
        flattened = preprocessor_grayscale.create_flattened_features(processed)

        # grayscale: 64 * 64 = 4,096
        assert flattened.shape == (10, 4096)
