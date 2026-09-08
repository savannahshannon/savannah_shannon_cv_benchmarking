#!/usr/bin/env python3
"""Simple test runner for the CV benchmarking package."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from savannah_shannon_cv_benchmarking.classical_models import ClassicalModelTrainer
from savannah_shannon_cv_benchmarking.neural_models import NeuralNetworkTrainer

def test_classical_trainer():
    """Test classical trainer."""
    print("=" * 60)
    print("Testing ClassicalModelTrainer...")
    print("=" * 60)
    X_train = np.random.randn(30, 12288)
    y_train = np.array([0]*15 + [1]*15)
    
    trainer = ClassicalModelTrainer()
    results = trainer.train_all_models(X_train, y_train)
    
    assert len(results) == 4, f"Expected 4 models, got {len(results)}"
    for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']:
        assert model_name in results, f"Missing model: {model_name}"
        assert 'model' in results[model_name], f"Missing 'model' key in {model_name}"
        assert 'train_time' in results[model_name], f"Missing 'train_time' key in {model_name}"
    
    print("✅ ClassicalModelTrainer test PASSED\n")
    return True

def test_neural_trainer():
    """Test neural trainer."""
    print("=" * 60)
    print("Testing NeuralNetworkTrainer...")
    print("=" * 60)
    X_train = np.random.randint(0, 256, (30, 64, 64, 3), dtype=np.uint8)
    y_train = np.array([0]*15 + [1]*15)
    
    trainer = NeuralNetworkTrainer(epochs=2)
    results = trainer.train_all_models(X_train, y_train)
    
    assert len(results) == 2, f"Expected 2 models, got {len(results)}"
    for model_name in ['Neural Network', 'Simple CNN']:
        assert model_name in results, f"Missing model: {model_name}"
        assert 'model' in results[model_name], f"Missing 'model' key in {model_name}"
        assert 'train_time' in results[model_name], f"Missing 'train_time' key in {model_name}"
    
    print("✅ NeuralNetworkTrainer test PASSED\n")
    return True

if __name__ == '__main__':
    try:
        test_classical_trainer()
        test_neural_trainer()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
