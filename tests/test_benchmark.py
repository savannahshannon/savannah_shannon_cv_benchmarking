"""
Integration tests for the complete benchmarking pipeline.
Tests the end-to-end workflow from data loading through evaluation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from savannah_shannon_cv_benchmarking import benchmark_image_classification

class TestBenchmarkPipeline:
    # test the complete benchmark pipeline

    @pytest.fixture
    def sample_dataset(self):
        # create a small sample dataset for testing
        np.random.seed(42)
        
        # 40 images, 2 classes, 32×32×3 (RGB)
        X = np.random.randint(0, 256, (40, 32, 32, 3), dtype=np.uint8)
        y = np.array([0]*20 + [1]*20)
        
        return X, y

    def test_benchmark_with_array_input(self, sample_dataset):
        # test benchmark with numPy array input
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        # verify return structure
        assert 'package_information' in results
        assert 'summary' in results
        assert 'best_model' in results
        assert 'dataset_information' in results
        assert 'split_information' in results
        assert 'output_files' in results
        
        # verify summary DataFrame
        assert isinstance(results['summary'], pd.DataFrame)
        assert len(results['summary']) == 6  # 6 models
        assert 'Model' in results['summary'].columns
        assert 'Accuracy' in results['summary'].columns
        assert 'Macro F1' in results['summary'].columns
        
        # verify best model
        assert isinstance(results['best_model'], str)
        assert results['best_model'] in results['summary']['Model'].values
        
        # verify dataset info
        assert results['dataset_information']['number_of_images'] == 40
        assert results['dataset_information']['number_of_classes'] == 2
        
        # verify split info
        assert results['split_information']['training_samples'] == 32  
        assert results['split_information']['testing_samples'] == 8  

    def test_benchmark_grayscale_mode(self, sample_dataset):
        # test benchmark in grayscale mode
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="grayscale",
                output_dir=tmpdir
            )
        
        # verify grayscale preprocessing
        assert results['dataset_information']['color_mode'] == 'grayscale'
        assert results['dataset_information']['image_shape'][2] == 1  # 1 channel

    def test_benchmark_output_files_created(self, sample_dataset):
        # test that all output files are created
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
            output_files = results['output_files']
        
            # verify files exist
            assert Path(output_files['summary_csv']).exists()
            assert Path(output_files['metrics_json']).exists()
            assert Path(output_files['config_json']).exists()

    def test_confusion_matrices_structure(self, sample_dataset):
        # test confusion matrices are properly formed
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        cm_dict = results['confusion_matrices']
        
        # should have 6 models
        assert len(cm_dict) == 6
        
        # each confusion matrix should be 2×2 (for 2 classes)
        for model_name, cm in cm_dict.items():
            assert isinstance(cm, list)
            cm_array = np.array(cm)
            assert cm_array.shape == (2, 2)

    def test_classification_reports_structure(self, sample_dataset):
        # test classification reports are properly formed
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        reports = results['classification_reports']
        
        assert len(reports) == 6
        
        # each report should have per-class metrics
        for model_name, report in reports.items():
            assert 'precision' in report or 'accuracy' in report

    def test_metrics_in_reasonable_range(self, sample_dataset):
        # test that computed metrics are in valid ranges
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        summary = results['summary']
        
        # all metrics should be between 0 and 1
        for col in ['Accuracy', 'Macro Precision', 'Macro Recall', 
                    'Macro F1', 'Weighted F1']:
            assert (summary[col] >= 0).all()
            assert (summary[col] <= 1).all()
        
        # training and inference times should be positive
        assert (summary['Training Time (s)'] > 0).all()
        assert (summary['Inference Time (ms/image)'] > 0).all()

    def test_summary_sorted_by_f1_score(self, sample_dataset):
        # test that summary is sorted by Macro F1 score (descending)
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        summary = results['summary']
        f1_scores = summary['Macro F1'].values
        
        # verify descending order
        assert (f1_scores[:-1] >= f1_scores[1:]).all()

    def test_reproducibility_with_seed(self):
        # test that results are reproducible with fixed random seed
        np.random.seed(42)
        X = np.random.randint(0, 256, (30, 32, 32, 3), dtype=np.uint8)
        y = np.array([0]*15 + [1]*15)
        
        with tempfile.TemporaryDirectory() as tmpdir1:
            results1 = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir1
            )
        
        # run again with same data
        with tempfile.TemporaryDirectory() as tmpdir2:
            results2 = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir2
            )
        
        # results should be identical
        assert results1['best_model'] == results2['best_model']
        assert np.allclose(
            results1['summary']['Macro F1'].values,
            results2['summary']['Macro F1'].values
        )

    def test_model_results_dictionary(self, sample_dataset):
        # test model results dictionary structure
        X, y = sample_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        model_results = results['model_results']
        
        assert len(model_results) == 6
        
        for model_name, metrics in model_results.items():
            assert 'accuracy' in metrics
            assert 'macro_f1' in metrics
            assert 'training_time' in metrics
            assert 'inference_time' in metrics


class TestDataValidation:
    # test dataset validation and error handling

    def test_insufficient_classes_raises_error(self):
        # test that dataset with only 1 class raises error
        X = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        y = np.zeros(20)  # Only class 0
        
        with pytest.raises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                benchmark_image_classification(
                    dataset=X,
                    dataset_type="array",
                    target_labels=y,
                    color_mode="rgb",
                    output_dir=tmpdir
                )

    def test_minimum_images_required(self):
        # test that minimum dataset size works
        X = np.random.randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = benchmark_image_classification(
                dataset=X,
                dataset_type="array",
                target_labels=y,
                color_mode="rgb",
                output_dir=tmpdir
            )
        
        assert results is not None
        assert 'summary' in results


class TestPreprocessing:
    # test image preprocessing
    def test_image_resizing_to_64x64(self):
        # test that all images are resized to 64×64
        from savannah_shannon_cv_benchmarking.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=64, color_mode='rgb')
        
        # test various input sizes
        sizes = [(32, 32), (128, 128), (100, 50), (50, 100)]
        
        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            batch = preprocessor.preprocess_batch(np.array([img]))
            
            assert batch.shape == (1, 64, 64, 3)

    def test_normalization_to_unit_range(self):
        # test that pixels are normalized to [0, 1]
        from savannah_shannon_cv_benchmarking.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=64, color_mode='rgb')
        
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        batch = preprocessor.preprocess_batch(np.array([img]))
        
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_grayscale_conversion(self):
        # test grayscale conversion produces single channel
        from savannah_shannon_cv_benchmarking.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=64, color_mode='grayscale')
        
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        batch = preprocessor.preprocess_batch(np.array([img]))
        
        assert batch.shape == (1, 64, 64, 1)


class TestModelTraining:
    # test individual model training
    def test_all_6_models_train(self):
        # test that all 6 models can be trained
        from savannah_shannon_cv_benchmarking.classical_models import ClassicalModelTrainer
        from savannah_shannon_cv_benchmarking.neural_models import NeuralNetworkTrainer
        
        np.random.seed(42)
        X_train = np.random.randn(30, 12288)
        X_test = np.random.randn(10, 12288)
        y_train = np.array([0]*15 + [1]*15)
        y_test = np.array([0]*5 + [1]*5)
        
        # classical models
        classical_trainer = ClassicalModelTrainer()
        classical_results = classical_trainer.train_all_models(
            X_train, y_train
        )
        
        assert len(classical_results) == 4
        for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']:
            assert model_name in classical_results
            assert 'model' in classical_results[model_name]
            assert 'train_time' in classical_results[model_name]
        
        # neural models
        X_train_img = np.random.randint(0, 256, (30, 64, 64, 3), dtype=np.uint8)
        X_test_img = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
        
        neural_trainer = NeuralNetworkTrainer(epochs=2)
        neural_results = neural_trainer.train_all_models(
            X_train_img, y_train
        )
        
        assert len(neural_results) == 2
        for model_name in ['Neural Network', 'Simple CNN']:
            assert model_name in neural_results
            assert 'model' in neural_results[model_name]
            assert 'train_time' in neural_results[model_name]


if __name__ == '__main__':
    # run tests: pytest tests/test_benchmark.py -v
    pytest.main([__file__, '-v'])