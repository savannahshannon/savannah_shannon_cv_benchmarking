"""
Main benchmark orchestration module.
Provides the public API for running complete image classification benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
import random
np.random.seed(42)
random.seed(42)
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Union, List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    import os
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
except ImportError:
    pass

from .data_loader import DataLoader
from .preprocessing import ImagePreprocessor
from .classical_models import ClassicalModelTrainer
from .neural_models import NeuralNetworkTrainer
from .evaluation import ModelEvaluator
from .visualization import BenchmarkVisualizer

def benchmark_image_classification(
    dataset: Union[str, np.ndarray, 'pd.DataFrame'],
    dataset_type: str,
    target_labels: Union[str, List, np.ndarray],
    color_mode: str,
    output_dir: str = 'benchmark_results'
) -> Dict:
    # Load an image dataset, train all 6 classifiers, and return a complete benchmark comparison.

    # Internal Constraints
    _IMAGE_SIZE = 64
    _RANDOM_SEED = 42
    _TRAIN_TEST_SPLIT = 0.20
    _EPOCHS = 20
    _BATCH_SIZE = 32

    np.random.seed(_RANDOM_SEED)
    random.seed(_RANDOM_SEED)
    
    print("CV Benchmarking Pipeline Started")

    # step 1: data loading
    print("Step 1: Loading dataset...")
    loader = DataLoader()

    try:
        # special handling for array format
        if dataset_type == "array":
            image_data = dataset
            labels = target_labels if isinstance(target_labels, np.ndarray) else np.array(target_labels)
            class_names = [str(i) for i in sorted(np.unique(labels))]
            load_info = {'format': 'array', 'n_images': len(image_data)}
        else:
            image_data, labels, class_names, load_info = loader.load(
                dataset, dataset_type
            )
        print(f"Loaded {len(labels)} images from {len(class_names)} classes")
        print(f"Classes: {class_names}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

    # step 2: preprocessing
    print("\nStep 2: Preprocessing images...")
    preprocessor = ImagePreprocessor(image_size=_IMAGE_SIZE, color_mode=color_mode)

    try:
        # preprocess images
        image_batch = preprocessor.preprocess_batch(image_data)
        print(f"Preprocessed images to shape {image_batch.shape}")
        print(f"Pixel range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")
    except Exception as e:
        raise Exception(f"Error during preprocessing: {str(e)}")

    # step 3: train-test split
    print("\nStep 3: Creating train-test split...")

    # for very small datasets, skip stratification if test set is smaller than number of classes
    # stratified split requires at least 1 sample of each class in both train and test sets
    use_stratify = True
    test_size_count = int(len(labels) * _TRAIN_TEST_SPLIT)

    if test_size_count < len(class_names):
        print("Warning: Small dataset detected, disabling stratification")
        use_stratify = False

    X_train, X_test, y_train, y_test = train_test_split(
        image_batch, labels,
        test_size=_TRAIN_TEST_SPLIT,
        stratify=labels if use_stratify else None,
        random_state=_RANDOM_SEED
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # compute class distributions for reporting
    train_distribution = {}
    test_distribution = {}
    for cls_idx, cls_name in enumerate(class_names):
        train_distribution[cls_name] = int(np.sum(y_train == cls_idx))
        test_distribution[cls_name] = int(np.sum(y_test == cls_idx))

    # step 4: model training
    print("\nStep 4: Training models...")

    # prepare data for different model types
    X_train_flat = preprocessor.create_flattened_features(X_train)
    X_test_flat = preprocessor.create_flattened_features(X_test)

    all_predictions = {}
    all_training_times = {}
    all_inference_times = {}

    # train classical models (LR, DT, RF, SVM)
    print("Training classical models...")
    classical_trainer = ClassicalModelTrainer(random_state=_RANDOM_SEED)
    classical_results = classical_trainer.train_all_models(X_train_flat, y_train)
    classical_predictions = classical_trainer.predict_all(X_test_flat)
    classical_times = classical_trainer.get_inference_times(X_test_flat)

    all_predictions.update(classical_predictions)
    all_training_times.update(classical_trainer.training_times)
    all_inference_times.update(classical_times)

    for model_name in classical_results.keys():
        print(f"  {model_name} trained")

    # train neural models (MLP, CNN)
    print("Training neural models...")
    neural_trainer = NeuralNetworkTrainer(
        random_state=_RANDOM_SEED,
        epochs=_EPOCHS,
        batch_size=_BATCH_SIZE
    )
    neural_results = neural_trainer.train_all_models(X_train, y_train)
    neural_predictions = neural_trainer.predict_all(X_test)
    neural_times = neural_trainer.get_inference_times(X_test)

    all_predictions.update(neural_predictions)
    all_training_times.update(neural_trainer.training_times)
    all_inference_times.update(neural_times)

    for model_name in neural_results.keys():
        print(f"  {model_name} trained")

    # step 5: eval
    print("\nStep 5: Evaluating models...")
    evaluator = ModelEvaluator()
    summary_df = evaluator.evaluate_all_predictions(
        all_predictions,
        y_test,
        all_training_times,
        all_inference_times,
        class_names
    )

    print("\nBenchmark Summary:")
    print(summary_df.to_string(index=False))

    best_model = summary_df.iloc[0]['Model']
    print(f"\nBest model: {best_model} (Macro F1: {summary_df.iloc[0]['Macro F1']:.4f})")

    # step 6: visualization
    print("\nStep 6: Generating visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organized output
    confusion_matrices_dir = output_dir / 'confusion_matrices'
    classification_reports_dir = output_dir / 'classification_reports'
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
    classification_reports_dir.mkdir(parents=True, exist_ok=True)

    visualizer = BenchmarkVisualizer(output_dir=str(output_dir))
    output_files = {}

    try:
        # class distribution
        visualizer.plot_class_distribution(labels, class_names)
        output_files['class_distribution'] = str(output_dir / "class_distribution.png")
        print("Class distribution plot")

        # split distribution
        visualizer.plot_split_distribution(len(y_train), len(y_test))
        output_files['split_distribution'] = str(output_dir / "split_distribution.png")
        print("Train-test split plot")

        # model comparison
        visualizer.plot_model_comparison(summary_df)
        output_files['model_comparison'] = str(output_dir / "model_comparison.png")
        print("Model comparison plot")

        # precision-recall
        visualizer.plot_precision_recall_comparison(summary_df)
        output_files['precision_recall'] = str(output_dir / "precision_recall_comparison.png")
        print("Precision-Recall plot")

        # confusion matrices
        confusion_matrices_paths = {}
        for model_name in summary_df['Model'].values:
            cm = evaluator.get_confusion_matrix(model_name)
            # Create a safe filename from model name
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()

            # save confusion matrix to confusion_matrices subdirectory
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'})
            plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

            cm_path = confusion_matrices_dir / f"confusion_matrix_{safe_name}.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()

            confusion_matrices_paths[model_name] = str(cm_path)
            print(f"Confusion matrix for {model_name}")
        output_files['confusion_matrices'] = confusion_matrices_paths
    except Exception as e:
        print(f"Warning: Could not generate all visualizations: {str(e)}")

    # save results
    print("\nSaving benchmark results...")

    # CSV
    summary_csv_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    output_files['summary_csv'] = str(summary_csv_path)
    print(f"Summary table: {summary_csv_path}")

    # save classification reports as CSV
    classification_reports_paths = {}
    for model_name in summary_df['Model'].values:
        report_dict = evaluator.get_classification_report(model_name)
        # convert classification report dict to DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        # create a safe filename from model name
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        report_csv_path = classification_reports_dir / f"classification_report_{safe_name}.csv"
        report_df.to_csv(report_csv_path)
        classification_reports_paths[model_name] = str(report_csv_path)
        print(f"Classification report saved: {report_csv_path}")
    output_files['classification_reports'] = classification_reports_paths

    # JSON metrics
    benchmark_metrics = {}
    for idx, row in summary_df.iterrows():
        model_name = row['Model']
        benchmark_metrics[model_name] = {
            'accuracy': float(row['Accuracy']),
            'macro_precision': float(row['Macro Precision']),
            'macro_recall': float(row['Macro Recall']),
            'macro_f1': float(row['Macro F1']),
            'weighted_f1': float(row['Weighted F1']),
            'train_time': float(row['Training Time (s)']),
            'inference_time': float(row['Inference Time (ms/image)'])
        }

    metrics_json_path = output_dir / "benchmark_metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(benchmark_metrics, f, indent=2)
    output_files['metrics_json'] = str(metrics_json_path)
    print(f"Benchmark metrics JSON: {metrics_json_path}")

    # run configuration
    run_config = {
        'image_size': _IMAGE_SIZE,
        'random_seed': _RANDOM_SEED,
        'color_mode': color_mode,
        'train_test_split': f"{1-_TRAIN_TEST_SPLIT:.0%} train, {_TRAIN_TEST_SPLIT:.0%} test",
        'epochs': _EPOCHS,
        'batch_size': _BATCH_SIZE,
        'models_trained': list(summary_df['Model'].values)
    }

    config_json_path = output_dir / "run_config.json"
    with open(config_json_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    output_files['config_json'] = str(config_json_path)
    print(f"Configuration JSON: {config_json_path}")

    # final results
    results = {
        'package_information': {
            'version': "1.0.0",
            'author': "Savannah Shannon",
            'name': "Savannah Shannon CV Benchmarking"
        },
        'summary': summary_df,
        'best_model': best_model,
        'dataset_information': {
            'dataset_type': dataset_type,
            'number_of_images': len(labels),
            'number_of_classes': len(class_names),
            'class_names': class_names,
            'color_mode': color_mode,
            'image_shape': [_IMAGE_SIZE, _IMAGE_SIZE, image_batch.shape[3]]
        },
        'split_information': {
            'training_samples': len(y_train),
            'testing_samples': len(y_test),
            'random_seed': _RANDOM_SEED,
            'train_distribution': train_distribution,
            'test_distribution': test_distribution
        },
        'model_results': {
            row['Model']: {
                'accuracy': float(row['Accuracy']),
                'macro_f1': float(row['Macro F1']),
                'training_time': float(row['Training Time (s)']),
                'inference_time': float(row['Inference Time (ms/image)'])
            }
            for _, row in summary_df.iterrows()
        },
        'confusion_matrices': {
            model_name: evaluator.get_confusion_matrix(model_name).tolist()
            for model_name in summary_df['Model'].values
        },
        'classification_reports': {
            model_name: evaluator.get_classification_report(model_name)
            for model_name in summary_df['Model'].values
        },
        'output_files': output_files
    }

    print("BENCHMARKING COMPLETE")
    print(f"\nResults saved to: {output_dir.absolute()}")

    return results