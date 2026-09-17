"""
CNN Benchmark Master Orchestration Script
Trains all 10 CNN architectures and compiles comprehensive benchmark results.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from .cnn_models import get_cnn_model, count_parameters
from .cnn_train import train_cnn_model
from .cnn_efficiency import (measure_inference_latency, measure_throughput,
                            measure_model_size, measure_gpu_memory, 
                            count_model_parameters)


class CNNBenchmark:
    # orchestrate CNN benchmark across multiple architectures
    
    # all 10 supported architectures
    ARCHITECTURES = [
        'resnet50',
        'resnet18',
        'densenet121',
        'efficientnet_b0',
        'mobilenet_v2',
        'vit_b_16',
        'convnext_tiny',
        'alexnet',
        'vgg16',
        'inception_v3'
    ]
    
    def __init__(self, 
                dataset_name: str = 'cifar10',
                checkpoint_dir: str = 'cnn_results/checkpoints',
                results_dir: str = 'cnn_results'):
        """
        initialize CNN benchmark.
        args:
            dataset_name: Name of dataset (cifar10, mnist, etc.)
            checkpoint_dir: Directory to save model checkpoints
            results_dir: Directory to save results and visualizations
        """
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
        
        self.results = {}
        
    def train_architecture(self, 
                        model_name: str,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: DataLoader,
                        epochs: int = 20,
                        learning_rate: float = 0.001,
                        num_classes: int = 10) -> dict:
        """
        train a single CNN architecture
        args:
            model_name: Name of architecture
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            num_classes: Number of classes   
        returns:
            dictionary with training results and metrics
        """
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # initialize model
        model = get_cnn_model(model_name, num_classes=num_classes, pretrained=True)
        model = model.to(self.device)
        
        # get parameter counts
        total_params, trainable_params = count_model_parameters(model)
        
        # train model
        history = train_cnn_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
            model_name=model_name,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # load best checkpoint
        checkpoint_path = f"{self.checkpoint_dir}/best_{model_name}.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        # evaluate on test set
        test_metrics = self._evaluate_model(model, test_loader, num_classes)
        
        # measure efficiency
        efficiency_metrics = self._measure_efficiency(model, test_loader, model_name)
        
        # compile results
        result = {
            'model': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision_macro': test_metrics['precision_macro'],
            'test_recall_macro': test_metrics['recall_macro'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_precision_weighted': test_metrics['precision_weighted'],
            'test_recall_weighted': test_metrics['recall_weighted'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'inference_latency_ms': efficiency_metrics['latency_ms'],
            'throughput_img_per_sec': efficiency_metrics['throughput'],
            'model_size_mb': efficiency_metrics['model_size_mb'],
            'gpu_memory_mb': efficiency_metrics['gpu_memory_mb'],
            'best_val_accuracy': max(history['val_acc']),
            'best_val_loss': min(history['val_loss']),
            'training_time_sec': sum(history['epoch_times']),
            'avg_epoch_time_sec': np.mean(history['epoch_times']),
            'confusion_matrix': test_metrics['confusion_matrix']
        }
        
        self.results[model_name] = result
        return result
    
    def _evaluate_model(self, model: nn.Module, 
                    test_loader: DataLoader,
                    num_classes: int) -> dict:
        # evaluate model on test set
        model.eval()
        
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                if isinstance(outputs, tuple): 
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = correct / total
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm
        }
    
    def _measure_efficiency(self, model: nn.Module,
                        test_loader: DataLoader,
                        model_name: str) -> dict:
        # measure efficiency metrics
        
        latency_ms = measure_inference_latency(model, test_loader, device=self.device, batch_size=1)
        throughput = measure_throughput(model, test_loader, device=self.device, batch_size=32)
        model_size_mb = measure_model_size(model, self.checkpoint_dir, model_name)
        gpu_memory_mb = measure_gpu_memory(model, test_loader, device=self.device, batch_size=32)
        
        return {
            'latency_ms': latency_ms,
            'throughput': throughput,
            'model_size_mb': model_size_mb,
            'gpu_memory_mb': gpu_memory_mb
        }
    
    def save_results_csv(self, output_file: str = 'combined_ml_cnn_benchmark_results.csv'):
        # save results to CSV
        
        # convert results to dataframe
        results_list = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Total_Parameters': metrics['total_params'],
                'Trainable_Parameters': metrics['trainable_params'],
                'Test_Accuracy': metrics['test_accuracy'],
                'Precision_Macro': metrics['test_precision_macro'],
                'Recall_Macro': metrics['test_recall_macro'],
                'F1_Macro': metrics['test_f1_macro'],
                'Precision_Weighted': metrics['test_precision_weighted'],
                'Recall_Weighted': metrics['test_recall_weighted'],
                'F1_Weighted': metrics['test_f1_weighted'],
                'Inference_Latency_ms': metrics['inference_latency_ms'],
                'Throughput_img_per_sec': metrics['throughput_img_per_sec'],
                'Model_Size_MB': metrics['model_size_mb'],
                'GPU_Memory_MB': metrics['gpu_memory_mb'],
                'Training_Time_sec': metrics['training_time_sec'],
                'Avg_Epoch_Time_sec': metrics['avg_epoch_time_sec']
            }
            results_list.append(row)
        
        df = pd.DataFrame(results_list)
        df = df.sort_values('Test_Accuracy', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return df
    
    def plot_accuracy_comparison(self):
        # plot accuracy comparison across architectures
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_accuracy'] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color='steelblue', alpha=0.7, edgecolor='black')
        
        # add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Architecture', fontsize=12, fontweight='bold')
        plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        plt.title('CNN Architectures - Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_metrics(self):
        # plot efficiency metrics (latency, throughput, model size)
        
        models = list(self.results.keys())
        latencies = [self.results[m]['inference_latency_ms'] for m in models]
        throughputs = [self.results[m]['throughput_img_per_sec'] for m in models]
        sizes = [self.results[m]['model_size_mb'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # latency
        axes[0].bar(models, latencies, color='coral', alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Latency (ms)', fontweight='bold')
        axes[0].set_title('Inference Latency', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # throughput
        axes[1].bar(models, throughputs, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Throughput (images/sec)', fontweight='bold')
        axes[1].set_title('Inference Throughput', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # model size
        axes[2].bar(models, sizes, color='lightyellow', alpha=0.7, edgecolor='black')
        axes[2].set_ylabel('Model Size (MB)', fontweight='bold')
        axes[2].set_title('Model Size', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/efficiency_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, num_classes: int = 10):
        # plot confusion matrices for all architectures
        
        for model_name, metrics in self.results.items():
            cm = metrics['confusion_matrix']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.ylabel('True Label', fontweight='bold')
            plt.title(f'Confusion Matrix - {model_name.upper()}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/confusion_matrices/{model_name}_cm.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
