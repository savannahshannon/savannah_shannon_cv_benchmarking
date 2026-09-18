"""
CNN Benchmarking Suite - Trains and evaluates CNN architectures on CIFAR-10.

Usage:
    python run_cnn_benchmark.py --model all --epochs 10
    python run_cnn_benchmark.py --model efficientnet_b0 --epochs 20
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ALL_MODELS = [
    'alexnet', 'vgg16', 'googlenet', 'resnet18', 'resnet50',
    'densenet121', 'mobilenet_v3_small', 'efficientnet_b0', 'convnext_tiny',
]


def get_model(name, num_classes=10):
    """Model factory."""
    if name == 'alexnet':
        m = models.alexnet(weights='DEFAULT')
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif name == 'vgg16':
        m = models.vgg16(weights='DEFAULT')
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif name == 'googlenet':
        m = models.googlenet(weights='DEFAULT', aux_logits=True)
        m.fc = nn.Linear(1024, num_classes)
    elif name == 'resnet18':
        m = models.resnet18(weights='DEFAULT')
        m.fc = nn.Linear(512, num_classes)
    elif name == 'resnet50':
        m = models.resnet50(weights='DEFAULT')
        m.fc = nn.Linear(2048, num_classes)
    elif name == 'densenet121':
        m = models.densenet121(weights='DEFAULT')
        m.classifier = nn.Linear(1024, num_classes)
    elif name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights='DEFAULT')
        m.classifier[3] = nn.Linear(1024, num_classes)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights='DEFAULT')
        m.classifier[1] = nn.Linear(1280, num_classes)
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights='DEFAULT')
        m.classifier[2] = nn.Linear(768, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def get_dataloaders(dataset='cifar10', batch_size=32):
    # load CIFAR-10 with 80/10/10 train/val/test split
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), norm
    ])
    train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train, val = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train_one_model(model_name, args, loaders):
    # train a single model, return metrics dict
    train_loader, val_loader, test_loader = loaders
    device = args.device
    
    print(f"\n{'='*70}\n[{model_name.upper()}] Loading model...\n{'='*70}")
    model = get_model(model_name, num_classes=10).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'epoch_time': [], 'lr': []}
    best_val_acc = 0
    training_start = time.time()
    
    for ep in range(args.epochs):
        ep_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            if model_name == 'googlenet':
                out = model(imgs)
                if isinstance(out, tuple): out = out[0]
            else:
                out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == lbls).sum().item()
            train_total += lbls.size(0)
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += criterion(out, lbls).item()
                val_correct += (out.argmax(1) == lbls).sum().item()
                val_total += lbls.size(0)
        
        train_acc, val_acc = train_correct/train_total, val_correct/val_total
        ep_time = time.time() - ep_start
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(ep_time)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'cnn_results/checkpoints/best_{model_name}.pt')
        
        print(f"  E{ep+1}/{args.epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | {ep_time:.1f}s")
        scheduler.step()
    
    training_time = time.time() - training_start
    
    # load best checkpoint and test
    model.load_state_dict(torch.load(f'cnn_results/checkpoints/best_{model_name}.pt'))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            y_true.extend(lbls.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    test_acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(f'cnn_results/confusion_matrices/{model_name}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title(f'{model_name} - Loss'); axes[0].legend()
    axes[1].plot(history['train_acc'], label='Train'); axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title(f'{model_name} - Accuracy'); axes[1].legend()
    plt.savefig(f'cnn_results/plots/{model_name}_curves.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # efficiency
    model_size_mb = os.path.getsize(f'cnn_results/checkpoints/best_{model_name}.pt') / 1e6
    sample = next(iter(test_loader))[0][:1].to(device)
    for _ in range(10): model(sample)  # warmup
    if device == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): model(sample)
    if device == 'cuda': torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 100 * 1000
    throughput = 1000 / latency_ms
    gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == 'cuda' else 0
    
    return {
        'Model': model_name,
        'Total_Parameters': total_params,
        'Trainable_Parameters': trainable_params,
        'Test_Accuracy': round(test_acc, 4),
        'Precision_Macro': round(prec_macro, 4),
        'Recall_Macro': round(rec_macro, 4),
        'F1_Macro': round(f1_macro, 4),
        'Precision_Weighted': round(prec_weighted, 4),
        'Recall_Weighted': round(rec_weighted, 4),
        'F1_Weighted': round(f1_weighted, 4),
        'Inference_Latency_ms': round(latency_ms, 2),
        'Throughput_img_per_sec': round(throughput, 2),
        'Model_Size_MB': round(model_size_mb, 2),
        'GPU_Memory_MB': round(gpu_mem, 2),
        'Training_Time_sec': round(training_time, 2),
        'Avg_Epoch_Time_sec': round(np.mean(history['epoch_time']), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--model', default='all', help="Model name or 'all'")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    for d in ['cnn_results/checkpoints', 'cnn_results/plots', 'cnn_results/confusion_matrices', 'cnn_results/logs']:
        os.makedirs(d, exist_ok=True)
    
    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]
    print(f"\n{'#'*70}\nCNN BENCHMARK: {len(models_to_run)} model(s) on {args.dataset}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | Device: {args.device}\n{'#'*70}")
    
    print("Loading data...")
    loaders = get_dataloaders(args.dataset, args.batch_size)
    print(f"✓ {len(loaders[0])} train, {len(loaders[1])} val, {len(loaders[2])} test batches\n")
    
    all_results = []
    for i, model_name in enumerate(models_to_run, 1):
        print(f"\n{'*'*70}\n[{i}/{len(models_to_run)}] {model_name.upper()}\n{'*'*70}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            result = train_one_model(model_name, args, loaders)
            all_results.append(result)
            print(f"✓ {model_name}: Test Acc = {result['Test_Accuracy']:.4f}")
        except Exception as e:
            print(f"✗ {model_name} FAILED: {e}")
            all_results.append({'Model': model_name, 'Test_Accuracy': 'ERROR', 'Error': str(e)[:100]})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(all_results)
    df.to_csv('combined_ml_cnn_benchmark_results.csv', index=False)
    print(f"\n{'#'*70}\nCOMPLETE - Results saved to combined_ml_cnn_benchmark_results.csv\n{'#'*70}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()