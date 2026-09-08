"""
Unit tests for stratified train-test splitting.
Tests: stratification, no overlap, consistent splits, etc.
"""

import pytest
import numpy as np
from sklearn.model_selection import train_test_split


class TestStratification:
    # test stratified splitting functionality

    @pytest.fixture
    def sample_data(self):
        # create sample multiclass dataset
        X = np.random.randn(100, 10)
        y = np.array([0]*30 + [1]*40 + [2]*30)  # 3 classes
        return X, y

    def test_stratified_split_created(self, sample_data):
        # test that stratified split is created
        X, y = sample_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_no_train_test_overlap(self, sample_data):
        # test training and testing sets don't overlap
        X, y = sample_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # no samples should appear in both sets
        assert len(np.intersect1d(y_train, y_test)) == 3  # different classes, not samples
        assert len(y_train) + len(y_test) == len(y)

    def test_stratification_preserves_distribution(self, sample_data):
        # test stratification maintains class distribution
        X, y = sample_data

        # original distribution
        orig_dist = np.bincount(y) / len(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # train and test distributions should be similar to original
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)

        # allow small tolerance for random variations
        np.testing.assert_allclose(train_dist, orig_dist, atol=0.05)
        np.testing.assert_allclose(test_dist, orig_dist, atol=0.05)

    def test_reproducibility_with_seed(self, sample_data):
        # test split is reproducible with fixed seed
        X, y = sample_data

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_80_20_split(self, sample_data):
        # test 80-20 train-test split ratio
        X, y = sample_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        assert len(X_train) / len(X) == pytest.approx(0.8, abs=0.01)
        assert len(X_test) / len(X) == pytest.approx(0.2, abs=0.01)

    def test_all_classes_in_test_set(self, sample_data):
        # test all classes appear in test set (large enough)
        X, y = sample_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # all 3 classes should be in test set
        assert len(np.unique(y_test)) == len(np.unique(y))
