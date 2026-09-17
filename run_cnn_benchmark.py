#!/usr/bin/env python3
"""
Main entry point for CNN benchmarking.
Supports training all 10 architectures or specific models.

Usage:
    python run_cnn_benchmark.py --dataset cifar10 --model all --epochs 20
    python run_cnn_benchmark.py --dataset cifar10 --model resnet50 --epochs 20
"""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from savannah_shannon_cv_benchmarking.data_loader import get_data_loaders
from savannah_shannon_cv_benchmarking.cnn_benchmark import CNNBenchmark


def main():
    parser = argparse.ArgumentParser(
        description='CNN Benchmarking Suite for ML Comparison Project'
    )
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'],
                       help='Dataset to use for benchmarking')
    
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'resnet50', 'resnet18', 'densenet121',
                               'efficientnet_b0', 'mobilenet_v2', 'vit_b_16',
                               'convnext_tiny', 'alexnet', 'vgg16', 'inception_v3'],
                       help='Model architecture to train (all for all architectures)')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"CNN Benchmarking Suite")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {'All 10 architectures' if args.model == 'all' else args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )
        print(f"✓ Data loaded successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}\n")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Initialize benchmark
    benchmark = CNNBenchmark(
        dataset_name=args.dataset,
        checkpoint_dir='cnn_results/checkpoints',
        results_dir='cnn_results'
    )
    
    # Determine which models to train
    models_to_train = CNNBenchmark.ARCHITECTURES if args.model == 'all' else [args.model]
    
    # Train models
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Training {model_name}...")
        try:
            result = benchmark.train_architecture(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_classes=10
            )
            print(f"✓ {model_name} completed - Test Acc: {result['test_accuracy']:.4f}")
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    try:
        benchmark.plot_accuracy_comparison()
        benchmark.plot_efficiency_metrics()
        benchmark.plot_confusion_matrices(num_classes=10)
        print("✓ Visualizations generated")
    except Exception as e:
        print(f"⚠ Some visualizations failed: {e}")
    
    # Save results
    print(f"Saving results...")
    try:
        df = benchmark.save_results_csv('combined_ml_cnn_benchmark_results.csv')
        print(f"✓ Results saved")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"Benchmark Complete!")
    print(f"{'='*60}")
    print(f"\nResults Summary (sorted by accuracy):")
    print(df[['Model', 'Test_Accuracy', 'Inference_Latency_ms', 
              'Model_Size_MB', 'Total_Parameters']].to_string(index=False))
    print(f"\nOutputs saved to:")
    print(f"  - Results: combined_ml_cnn_benchmark_results.csv")
    print(f"  - Plots: cnn_results/plots/")
    print(f"  - Confusion Matrices: cnn_results/confusion_matrices/")
    print(f"  - Checkpoints: cnn_results/checkpoints/")


if __name__ == '__main__':
    main()
