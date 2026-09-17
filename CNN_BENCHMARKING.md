# CNN Benchmarking Extension

This document describes the CNN benchmarking extension to the ML model benchmarking project. It extends the traditional ML benchmarking work by evaluating 10 state-of-the-art CNN architectures using transfer learning.

## Overview

The CNN benchmarking extension provides:

- **10 Pre-trained Architectures**: ResNet50, ResNet18, DenseNet121, EfficientNetB0, MobileNetV2, ViT-B16, ConvNeXt-Tiny, AlexNet, VGG16, Inception-V3
- **Unified Training Loop**: Consistent training interface across all architectures
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score (macro and weighted)
- **Efficiency Metrics**: Inference latency, throughput, model size, GPU memory, parameter counts
- **Automatic Checkpointing**: Saves best model state for each architecture
- **Rich Visualizations**: Accuracy comparison, efficiency metrics plots, confusion matrices
- **CSV Results**: Combined benchmark table for easy comparison with traditional ML models

## Project Structure

```
savannah_shannon_cv_benchmarking/
├── src/savannah_shannon_cv_benchmarking/
│   ├── cnn_models.py              # Model factory for all 10 architectures
│   ├── cnn_train.py               # Unified training loop
│   ├── cnn_efficiency.py          # Efficiency measurement utilities
│   ├── cnn_benchmark.py           # Master orchestration script
│   ├── (existing ML benchmarking files...)
│   └── data_loader.py             # Data loading (shared with ML models)
├── cnn_results/                   # NEW: Results and checkpoints
│   ├── checkpoints/               # Saved model states (best_*.pt)
│   ├── plots/                     # Comparison visualizations
│   └── confusion_matrices/        # Per-architecture confusion matrices
├── run_cnn_benchmark.py           # Main entry point (NEW)
├── CNN_BENCHMARKING.md            # This file
└── combined_ml_cnn_benchmark_results.csv  # Final results table
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas scikit-learn matplotlib seaborn
```

### 2. Train All CNN Architectures

```bash
# Train all 10 architectures on CIFAR-10 with default settings
python run_cnn_benchmark.py --dataset cifar10 --model all --epochs 20

# Or on MNIST
python run_cnn_benchmark.py --dataset mnist --model all --epochs 20
```

### 3. Train a Single Architecture

```bash
# Train only ResNet50
python run_cnn_benchmark.py --dataset cifar10 --model resnet50 --epochs 20
```

### 4. Custom Training Settings

```bash
python run_cnn_benchmark.py \
    --dataset cifar10 \
    --model all \
    --epochs 30 \
    --batch-size 256 \
    --learning-rate 0.0005 \
    --device cuda
```

## Available Architectures

| Architecture | Type | Parameters | Latency | Speed |
|---|---|---|---|---|
| ResNet50 | CNN | ~25.5M | Medium | Fast |
| ResNet18 | CNN | ~11.2M | Low | Very Fast |
| DenseNet121 | CNN | ~7.9M | Low | Very Fast |
| EfficientNetB0 | CNN | ~5.3M | Low | Very Fast |
| MobileNetV2 | CNN | ~3.5M | Very Low | Fastest |
| ViT-B16 | Transformer | ~86M | High | Slow |
| ConvNeXt-Tiny | Modern CNN | ~28M | Medium | Fast |
| AlexNet | Historical CNN | ~61M | High | Slow |
| VGG16 | CNN | ~138M | Very High | Very Slow |
| Inception-V3 | CNN | ~27M | Medium | Fast |

## Output Files

### Results Table: `combined_ml_cnn_benchmark_results.csv`

Contains comprehensive benchmark results:
- **Model**: Architecture name
- **Total_Parameters**: Total model parameters
- **Trainable_Parameters**: Trainable parameters (typically all for fine-tuning)
- **Test_Accuracy**: Accuracy on test set
- **Precision/Recall/F1**: Macro and weighted variants
- **Inference_Latency_ms**: Single-image inference time (ms)
- **Throughput_img_per_sec**: Batch inference throughput (images/sec)
- **Model_Size_MB**: Disk size of saved checkpoint (MB)
- **GPU_Memory_MB**: Peak GPU memory during inference (MB)
- **Training_Time_sec**: Total training time across all epochs
- **Avg_Epoch_Time_sec**: Average time per epoch

### Visualizations

**cnn_results/plots/**
- `accuracy_comparison.png`: Bar chart of test accuracies
- `efficiency_metrics.png`: Three-panel plot of latency, throughput, model size

**cnn_results/confusion_matrices/**
- `{model_name}_cm.png`: Per-class confusion matrix for each architecture

### Model Checkpoints

**cnn_results/checkpoints/**
- `best_{model_name}.pt`: Saved state dict for best model of each architecture

## Key Implementation Details

### Model Factory Pattern

All models are instantiated via `get_cnn_model()` which:
1. Loads pretrained weights from torchvision
2. Replaces the final classification layer for 10 classes
3. Returns model ready for training

Example:
```python
from cnn_models import get_cnn_model

model = get_cnn_model('resnet50', num_classes=10, pretrained=True)
```

### Unified Training Loop

The `train_cnn_model()` function:
- Uses AdamW optimizer with cosine annealing learning rate scheduler
- Computes CrossEntropyLoss on training batches
- Validates on validation set each epoch
- Saves checkpoint when validation accuracy improves
- Handles Inception-V3's auxiliary outputs gracefully
- Returns history dict with losses, accuracies, and timing

### Efficiency Measurements

**Inference Latency** (`measure_inference_latency`)
- Measures end-to-end inference time for single images
- Includes GPU synchronization for accurate timing
- Averages over 100 batches after warm-up

**Throughput** (`measure_throughput`)
- Measures batch inference speed
- Standard batch size of 32 images
- Reports images per second

**Model Size** (`measure_model_size`)
- Reads saved checkpoint file size in MB
- Fallbacks to state_dict parameter count estimation

**GPU Memory** (`measure_gpu_memory`)
- Uses torch.cuda.max_memory_allocated()
- Measures peak memory during inference batch
- Returns 0.0 for CPU-only training

## Benchmark Workflow

1. **Load Data**: Uses existing `get_data_loaders()` with fixed SEED=42 splits
2. **Initialize Model**: Creates architecture with pretrained weights
3. **Train**: Runs unified training loop for specified epochs
4. **Save Best Checkpoint**: Persists best model state
5. **Evaluate**: Computes accuracy, precision, recall, F1, confusion matrix
6. **Measure Efficiency**: Benchmarks inference speed, memory, model size
7. **Compile Results**: Aggregates metrics into structured dictionary
8. **Visualize**: Generates comparison plots and confusion matrices
9. **Export**: Saves results to CSV for analysis

## Integration with Traditional ML Benchmarking

The CNN results are designed to be compared with your existing ML benchmarks:

**Existing ML Models** (from prior benchmarking):
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Neural Network (MLP)
- Simple CNN

**New CNN Models** (this extension):
- ResNet50, ResNet18
- DenseNet121
- EfficientNetB0
- MobileNetV2
- ViT-B16
- ConvNeXt-Tiny
- AlexNet
- VGG16
- Inception-V3

The combined results table (`combined_ml_cnn_benchmark_results.csv`) allows direct comparison across all 16+ models on:
- Accuracy metrics
- Efficiency (speed, size, memory)
- Parameter counts
- Training time

## Performance Tips

### For Faster Training
- Use smaller models: `mobilenet_v2`, `efficientnet_b0`, `resnet18`
- Reduce epochs: `--epochs 10`
- Increase batch size: `--batch-size 256` (if GPU memory allows)

### For Best Accuracy
- Use larger models: `vit_b_16`, `convnext_tiny`, `resnet50`
- Increase epochs: `--epochs 30` or `--epochs 50`
- Use smaller learning rate: `--learning-rate 0.0001`

### For Edge Deployment
- Focus on MobileNetV2, EfficientNetB0 (small, fast)
- Check inference_latency and model_size metrics
- Consider throughput for batch processing

## Troubleshooting

**Out of Memory (OOM)**
- Reduce batch size: `--batch-size 64`
- Use smaller model: `--model mobilenet_v2`
- Use CPU: `--device cpu`

**Slow Training**
- Use GPU if available
- Reduce epochs: `--epochs 5`
- Try lighter architecture: EfficientNetB0, MobileNetV2

**Poor Accuracy on Small Epochs**
- Increase epochs: `--epochs 50`
- Fine-tune learning rate: `--learning-rate 0.0005`
- Check data loading with traditional ML baseline

## Advanced: Custom Architecture or Different Dataset

To extend the benchmark:

1. **Add Architecture**: Update `ARCHITECTURES` list and `get_cnn_model()` in `cnn_models.py`
2. **Change Dataset**: Pass different `dataset_name` to `get_data_loaders()` and `CNNBenchmark()`
3. **Custom Metrics**: Extend evaluation in `CNNBenchmark._evaluate_model()`

## References

- Pretrained models: [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- Transfer Learning: [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Efficiency Benchmarking: [fvcore](https://github.com/facebookresearch/fvcore)

## Author Notes

This benchmarking suite demonstrates:
- **Architecture Evolution**: From AlexNet → VGG → ResNet → DenseNet → Transformers
- **Efficiency vs Accuracy Trade-offs**: Smaller models are faster but less accurate
- **Transfer Learning Power**: Pretrained models vastly outperform training from scratch
- **Deployment Considerations**: Model size and latency matter for real-world applications

---

Generated as part of the ML Model Benchmarking Project
