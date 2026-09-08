"""
Unit tests for model training functionality.
Tests: all 6 models train and predict correctly.
"""

import pytest
import numpy as np
from savannah_shannon_cv_benchmarking.classical_models import ClassicalModelTrainer
from savannah_shannon_cv_benchmarking.neural_models import NeuralNetworkTrainer


class TestClassicalModels:
    # test classical machine learning models

    @pytest.fixture
    def sample_training_data(self):
        # create sample training data
        X_train = np.random.randn(50, 100)
        y_train = np.array([0]*25 + [1]*25)
        X_test = np.random.randn(10, 100)
        return X_train, y_train, X_test

    def test_all_classical_models_train(self, sample_training_data):
        # test all 4 classical models train without error
        X_train, y_train, X_test = sample_training_data

        trainer = ClassicalModelTrainer(random_state=42)
        results = trainer.train_all_models(X_train, y_train)

        # should have 4 classical models
        assert len(results) == 4
        assert "Logistic Regression" in results
        assert "Decision Tree" in results
        assert "Random Forest" in results
        assert "SVM" in results

    def test_classical_model_predictions(self, sample_training_data):
        # test classical models make predictions
        X_train, y_train, X_test = sample_training_data

        trainer = ClassicalModelTrainer(random_state=42)
        trainer.train_all_models(X_train, y_train)
        predictions = trainer.predict_all(X_test)

        # should predict for all samples
        assert len(predictions["Logistic Regression"]) == len(X_test)
        assert len(predictions["Random Forest"]) == len(X_test)

    def test_classical_models_return_probabilities(self, sample_training_data):
        # test predictions are valid class labels
        X_train, y_train, X_test = sample_training_data

        trainer = ClassicalModelTrainer(random_state=42)
        trainer.train_all_models(X_train, y_train)
        predictions = trainer.predict_all(X_test)

        # all predictions should be valid class labels (0 or 1)
        for model_name, preds in predictions.items():
            assert all(p in [0, 1] for p in preds)


class TestNeuralModels:
    @pytest.fixture
    def sample_neural_data(self):
        """Create sample image training data."""
        X_train = np.random.randint(0, 256, (30, 32, 32, 3), dtype=np.uint8) / 255.0
        y_train = np.array([0]*15 + [1]*15)
        X_test = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8) / 255.0
        return X_train, y_train, X_test

    def test_both_neural_models_train(self, sample_neural_data):
        X_train, y_train, X_test = sample_neural_data

        trainer = NeuralNetworkTrainer(random_state=42, epochs=1, batch_size=8)
        results = trainer.train_all_models(X_train, y_train)

        # should have 2 neural models
        assert len(results) == 2
        assert "Neural Network" in results
        assert "Simple CNN" in results

    def test_neural_model_predictions(self, sample_neural_data):
        # test neural models make predictions
        X_train, y_train, X_test = sample_neural_data

        trainer = NeuralNetworkTrainer(random_state=42, epochs=1, batch_size=8)
        trainer.train_all_models(X_train, y_train)
        predictions = trainer.predict_all(X_test)

        # should predict for all samples
        assert len(predictions["Neural Network"]) == len(X_test)
        assert len(predictions["Simple CNN"]) == len(X_test)

    def test_neural_predictions_valid(self, sample_neural_data):
        # test neural network predictions are valid class labels
        X_train, y_train, X_test = sample_neural_data

        trainer = NeuralNetworkTrainer(random_state=42, epochs=1, batch_size=8)
        trainer.train_all_models(X_train, y_train)
        predictions = trainer.predict_all(X_test)

        # all predictions should be valid class labels
        for model_name, preds in predictions.items():
            assert all(p in [0, 1] for p in preds)


class TestModelConsistency:
    # test model training consistency across runs

    @pytest.fixture
    def sample_data(self):
        X_train = np.random.RandomState(42).randn(50, 100)
        y_train = np.array([0]*25 + [1]*25)
        X_test = np.random.RandomState(43).randn(10, 100)
        return X_train, y_train, X_test

    def test_classical_reproducibility(self, sample_data):
        # test classical models produce same results with same seed
        X_train, y_train, X_test = sample_data

        trainer1 = ClassicalModelTrainer(random_state=42)
        trainer1.train_all_models(X_train, y_train)
        preds1 = trainer1.predict_all(X_test)

        trainer2 = ClassicalModelTrainer(random_state=42)
        trainer2.train_all_models(X_train, y_train)
        preds2 = trainer2.predict_all(X_test)

        # predictions should be identical
        for model in ["Logistic Regression", "Random Forest"]:
            np.testing.assert_array_equal(preds1[model], preds2[model])
