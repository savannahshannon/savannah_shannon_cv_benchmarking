"""
Automated benchmark orchestrator: trains all CNN architectures sequentially.
Handles memory management, error recovery, and result aggregation.

Usage:
    python run_all_models.py                    # Train all models
    python run_all_models.py --epochs 10        # Custom epochs
    python run_all_models.py --skip resnet50    # Skip specific models
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
import pandas as pd

# all 10 CNN architectures 
ALL_MODELS = [
    'alexnet',           # 2012 - historical baseline
    'vgg16',             # 2014 - deep sequential
    'googlenet',         # 2014 - inception modules
    'resnet18',          # 2015 - residual (small)
    'resnet50',          # 2015 - residual (large)
    'densenet121',       # 2017 - dense connectivity
    'mobilenet_v3_small', # 2019 - efficient (mobile)
    'efficientnet_b0',   # 2019 - compound scaling
    'convnext_tiny',     # 2022 - modernized
    # YOLO handled separately (uses different framework)
]


def clear_gpu_memory():
    # free GPU memory between models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  GPU free: {free_mem:.2f} GB")


def train_model(model_name, epochs, batch_size, dataset='cifar10'):
    # run training for a single model via the existing benchmark script
    cmd = [
        sys.executable, 'run_cnn_benchmark.py',
        '--dataset', dataset,
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train_yolo_classification(epochs=5, batch_size=32):
    # train YOLO in classification mode (uses ultralytics)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Installing ultralytics...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'])
        from ultralytics import YOLO
    
    model = YOLO('yolov8n-cls.pt')
    results = model.train(
        data='cifar10',
        epochs=epochs,
        imgsz=224,
        batch=batch_size,
        project='cnn_results/yolo',
        name='cifar10_run',
        exist_ok=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description='Train all CNN architectures')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs per model (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32 for memory safety)')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Model names to skip')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only train these specific models')
    parser.add_argument('--include-yolo', action='store_true',
                        help='Also train YOLO classification')
    args = parser.parse_args()
    
    # determine which models to run
    if args.only:
        models = args.only
    else:
        models = [m for m in ALL_MODELS if m not in args.skip]
    
    print(f"\n{'='*70}")
    print(f"AUTOMATED CNN BENCHMARK PIPELINE")
    print(f"{'='*70}")
    print(f"Dataset:      {args.dataset}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Device:       {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"Models:       {len(models)} architectures")
    for m in models:
        print(f"                - {m}")
    if args.include_yolo:
        print(f"                - yolo_classification")
    print(f"Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # ensure output directories exist
    os.makedirs('cnn_results/checkpoints', exist_ok=True)
    os.makedirs('cnn_results/plots', exist_ok=True)
    os.makedirs('cnn_results/confusion_matrices', exist_ok=True)
    os.makedirs('cnn_results/logs', exist_ok=True)
    
    # track results
    log = []
    pipeline_start = time.time()
    
    for idx, model_name in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"[{idx}/{len(models)}] {model_name.upper()}")
        print(f"{'#'*70}")
        
        clear_gpu_memory()
        model_start = time.time()
        
        try:
            success = train_model(
                model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dataset=args.dataset,
            )
            elapsed = time.time() - model_start
            status = 'SUCCESS' if success else 'FAILED'
            print(f"\n[{idx}/{len(models)}] {model_name}: {status} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - model_start
            status = f'ERROR: {str(e)[:50]}'
            print(f"\n[{idx}/{len(models)}] {model_name}: {status}")
        
        log.append({
            'model': model_name,
            'status': status,
            'elapsed_sec': round(elapsed, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        
        # clear memory before next model
        clear_gpu_memory()
    
    # optional YOLO training
    if args.include_yolo:
        print(f"\n{'#'*70}")
        print(f"[BONUS] YOLO CLASSIFICATION")
        print(f"{'#'*70}")
        clear_gpu_memory()
        yolo_start = time.time()
        try:
            train_yolo_classification(epochs=args.epochs, batch_size=args.batch_size)
            log.append({
                'model': 'yolo_classification',
                'status': 'SUCCESS',
                'elapsed_sec': round(time.time() - yolo_start, 1),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        except Exception as e:
            log.append({
                'model': 'yolo_classification',
                'status': f'ERROR: {str(e)[:50]}',
                'elapsed_sec': round(time.time() - yolo_start, 1),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
    
    # final summary
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    print(f"\nPer-model status:")
    for entry in log:
        symbol = '✓' if entry['status'] == 'SUCCESS' else '✗'
        print(f"  {symbol} {entry['model']:25} {entry['status']:20} {entry['elapsed_sec']:.1f}s")
    
    # save log
    log_df = pd.DataFrame(log)
    log_path = f"cnn_results/logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nPipeline log saved: {log_path}")
    
    # success count
    success_count = sum(1 for e in log if e['status'] == 'SUCCESS')
    print(f"\nSuccessful: {success_count}/{len(log)} models")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()