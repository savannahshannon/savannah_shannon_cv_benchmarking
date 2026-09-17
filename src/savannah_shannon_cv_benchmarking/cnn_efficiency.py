"""
CNN Efficiency Measurements - Measure inference speed, model size, GPU memory, and parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
from importlib import import_module
from typing import Dict, Tuple


def measure_inference_latency(model: nn.Module,
                            test_loader: DataLoader,
                            device: str = 'cuda',
                            batch_size: int = 1) -> float:
    """
    measure average inference latency in ms
    args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to measure on
        batch_size: Batch size for inference
    returns:
        avg latency in ms
    """
    model.eval()
    model = model.to(device)
    
    total_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)[:batch_size]
            _ = model(images)
            break
    
    # measurement
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)[:batch_size]
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(images)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            total_time += time.time() - start
            num_batches += 1
            
            if num_batches >= 100:  # use 100 batches for stable measurement
                break
    
    avg_latency_ms = (total_time / num_batches) * 1000
    return avg_latency_ms


def measure_throughput(model: nn.Module,
                    test_loader: DataLoader,
                    device: str = 'cuda',
                    batch_size: int = 32) -> float:
    """
    measure inference throughput in images per second.
    args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to measure on
        batch_size: Batch size for inference   
    returns:
        throughput in images per second
    """
    model.eval()
    model = model.to(device)
    
    total_time = 0
    total_images = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)[:batch_size]
            _ = model(images)
            break
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)[:batch_size]
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(images)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            total_time += time.time() - start
            total_images += images.size(0)
            
            if total_images >= 3200:  # Measure 3200 images
                break
    
    throughput = total_images / total_time
    return throughput


def measure_model_size(model: nn.Module,
                    checkpoint_dir: str = 'cnn_results/checkpoints',
                    model_name: str = 'resnet50') -> float:
    """
    measure model size on disk in MB.
    args:
        model: PyTorch model
        checkpoint_dir: Directory where checkpoint is saved
        model_name: Model name for checkpoint path
        
    returns:
        model size in MB
    """
    checkpoint_path = f"{checkpoint_dir}/best_{model_name}.pt"
    
    if os.path.exists(checkpoint_path):
        size_bytes = os.path.getsize(checkpoint_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    else:
        # estimate from state_dict
        state_dict = model.state_dict()
        size_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())
        size_mb = size_bytes / (1024 * 1024)
        return size_mb


def measure_gpu_memory(model: nn.Module,
                    test_loader: DataLoader,
                    device: str = 'cuda',
                    batch_size: int = 32) -> float:
    """
    measure peak GPU memory usage in MB.
    args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to measure on
        batch_size: Batch size for inference
        
    returns:
        peak GPU memory in MB
    """
    if device != 'cuda':
        return 0.0
    
    model.eval()
    model = model.to(device)
    
    # clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)[:batch_size]
            _ = model(images)
            break
    
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    
    return peak_memory_mb


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    count total and trainable parameters.
    args:
        model: PyTorch model   
    returns:
        (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_flops(model: nn.Module,
                input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> int:
    """
    estimate FLOPs (Floating Point Operations).
    
    Note: This is a rough estimate. For exact FLOPs, use specialized tools like fvcore.
    
    args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        
    returns:
        estimated FLOPs
    """
    try:
        FlopCountAnalysis = import_module('fvcore.nn').FlopCountAnalysis
        model.eval()
        input_tensor = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, input_tensor).total()
        return flops
    except ImportError:
        # fallback: estimate from parameters
        total_params, _ = count_model_parameters(model)
        return total_params * 2  # rough estimate
