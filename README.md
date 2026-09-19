# Savannah Shannon CV Benchmarking Suite

**Integrated benchmarking library** comparing traditional ML classifiers with 10 modern CNN architectures on CIFAR-10. This project spans two assignments:

- **Assignment 2:** Reusable Python package for classical ML and simple neural benchmarks
- **Assignment 3 (Extension):** 10 pretrained CNN architectures with full efficiency profiling

Full evolution from Logistic Regression → EfficientNet-B0 in a single reproducible pipeline.

---

## Key Results

| Category | Best Model | Accuracy |
|----------|-----------|----------|
| Traditional ML (CIFAR-10, 1.5K subset) | SVM | 38.0% |
| Neural Baseline (CIFAR-10, 1.5K subset) | Simple CNN | 39.0% |
| **Deep CNN (Full CIFAR-10, 60K)** | **EfficientNet-B0** | **96.34%** |
| Most Efficient CNN | MobileNetV3-Small | 95.44% @ 1.53M params |
| Smallest CNN | YOLO Classification | 3 MB |

---

## What's Included

### Traditional ML & Baselines (Assignment 2)
- **Classical:** Logistic Regression, Decision Tree, Random Forest, SVM
- **Neural:** Fully Connected NN, Simple CNN

### Deep CNN Architectures (Assignment 3)
- **AlexNet** (2012) — Historical baseline
- **VGG16** (2014) — Deep sequential
- **GoogLeNet / Inception** (2014) — Multi-branch
- **ResNet18 / ResNet50** (2015) — Residual connections
- **DenseNet121** (2017) — Dense connectivity
- **MobileNetV3-Small** (2019) — Mobile-efficient
- **EfficientNet-B0** (2019) — Compound scaling
- **ConvNeXt-Tiny** (2022) — Modernized convnet
- **YOLO Classification** — Compact modern classifier

---

## Installation

Install from source:
```bash
git clone https://github.com/savannahshannon/savannah_shannon_cv_benchmarking.git
cd savannah_shannon_cv_benchmarking
pip install -r requirements.txt
```

Or install the Assignment 2 package from PyPI:
```bash
pip install Savannah_Shannon_CV_Benchmarking
```

---

## Quick Start

### Part 1: Traditional ML Benchmark (Assignment 2)

```python
from savannah_shannon_cv_benchmarking import benchmark_image_classification

results = benchmark_image_classification(
    dataset="path/to/your/dataset",
    dataset_type="folder",
    target_labels=["class1", "class2"],
    color_mode="rgb"
)
print(results['summary'])
print(f"Best model: {results['best_model']}")
```

### Part 2: CNN Benchmark (Assignment 3 Extension)

```bash
# Train all 9 CNN architectures on CIFAR-10
python run_cnn_benchmark.py --dataset cifar10 --model all --epochs 20 --batch-size 32

# Train a single architecture
python run_cnn_benchmark.py --dataset cifar10 --model efficientnet_b0 --epochs 20

# Generate all 8 comparison plots
python generate_plots.py
```

---

## Full Results

### Traditional ML on CIFAR-10 (1,500 image subset)

| Model | Accuracy | Macro F1 | Training Time | Inference (ms/img) |
|-------|----------|----------|---------------|---------------------|
| Simple CNN | 39.0% | 0.3810 | 11.94s | 0.203 |
| SVM | 38.0% | 0.3700 | 3.23s | 5.365 |
| Random Forest | 34.7% | 0.3374 | 0.84s | 0.069 |
| Logistic Regression | 25.3% | 0.2496 | 4.02s | 0.031 |
| Decision Tree | 19.0% | 0.1851 | 6.00s | 0.023 |
| Neural Network (FC) | 10.0% | 0.0182 | 2.91s | 0.102 |

### Deep CNN on Full CIFAR-10 (60,000 images, 20 epochs)

| Rank | Model | Params (M) | Test Acc | F1 Macro | Size (MB) | Latency (ms) |
|------|-------|-----------|----------|----------|-----------|--------------|
| 1 | EfficientNet-B0 | 4.02 | **96.34%** | 0.9634 | 16.4 | 10.52 |
| 2 | ConvNeXt-Tiny | 27.83 | 95.99% | 0.9598 | 111.4 | 6.78 |
| 3 | MobileNetV3-Small | 1.53 | 95.44% | 0.9544 | 6.3 | 6.70 |
| 4 | GoogLeNet | 11.99 | 95.39% | 0.9539 | 48.2 | 8.10 |
| 5 | DenseNet121 | 6.96 | 95.27% | 0.9526 | 28.5 | 17.83 |
| 6 | YOLO Classification | 1.45 | 95.10% | — | 3.0 | — |
| 7 | ResNet50 | 23.53 | 94.81% | 0.9481 | 94.4 | 6.88 |
| 8 | ResNet18 | 11.18 | 93.98% | 0.9397 | 44.8 | 2.66 |
| 9 | VGG16 | 134.30 | 77.07% | 0.7703 | 537.2 | 9.74 |
| 10 | AlexNet | 57.04 | 75.94% | 0.7594 | 228.2 | 2.10 |

**Note on Comparison:** Assignment 2 used a 1,500-image subset (150 per class); Assignment 3 uses full CIFAR-10 (60,000 images). Traditional ML numbers reflect that reduced-data setting. Even so, the accuracy gap (39% → 96%) demonstrates the dramatic value of pretrained deep architectures on natural images.

**Key Insights:**
- **Pretrained CNNs achieve 96.34% vs. 39% for Simple CNN** — transfer learning + scale wins
- **Modern > Old:** Post-2014 architectures dominate; AlexNet and VGG16 plateau at 75-77%
- **Parameter efficiency matters:** MobileNetV3-Small beats ResNet50 with 15× fewer parameters
- **Consistent errors:** Every model confuses Cat ↔ Dog (semantic similarity limit)

---

## Environment

**Hardware:**
- GPU: NVIDIA Tesla T4 (Google Colab, 14.56 GB VRAM)
- CPU: Intel Xeon @ 2.20 GHz
- RAM: 12 GB
- OS: Ubuntu 22.04

**Software:**
- Python: 3.13
- PyTorch: 2.11.0+cu128
- torchvision: 0.20+
- CUDA: 12.8
- Ultralytics: 8.4.155 (YOLO)
- scikit-learn, TensorFlow (Assignment 2 models)

---

## Reproducibility

- **Random seed:** `SEED = 42` (numpy + torch)
- **Split:** 80/10/10 train/val/test (CNN), 80/20 (Assignment 2)
- **CNN Optimizer:** AdamW (lr=0.001)
- **Scheduler:** CosineAnnealingLR
- **Batch size:** 32
- **Input resolution:** 224×224 (CNN), 64×64 (Assignment 2)
- **Normalization:** ImageNet stats for CNN

---

## Repository Structure
savannah_shannon_cv_benchmarking/
├── run_cnn_benchmark.py # Part 2: CNN benchmark entry
├── generate_plots.py # Part 2: comparison plot generator
├── src/savannah_shannon_cv_benchmarking/
│ ├── init.py
│ ├── benchmark_image_classification.py # Part 1: main orchestration
│ ├── classical_models.py # Part 1: sklearn models
│ ├── neural_models.py # Part 1: TF/Keras models
│ ├── evaluation.py # Part 1: metrics
│ ├── visualization.py # Part 1: plots
│ ├── preprocessing.py # Part 1: image preprocessing
│ ├── data_loader.py # Both parts: data loading
│ ├── cnn_models.py # Part 2: CNN model factory
│ ├── cnn_train.py # Part 2: training loop
│ ├── cnn_efficiency.py # Part 2: latency/memory
│ └── cnn_benchmark.py # Part 2: orchestration
├── cnn_results/
│ ├── checkpoints/ # Best CNN weights (.pt)
│ ├── plots/ # Training curves + 8 comparison plots
│ └── confusion_matrices/ # Per-model confusion matrices
├── examples/ # Assignment 2 example scripts
├── tests/ # pytest suite
├── combined_ml_cnn_benchmark_results.csv # Master results table
├── benchmark_summary.csv # Assignment 2 results
├── ANALYSIS.md # Rankings + architecture evolution
├── requirements.txt
├── pyproject.toml
└── README.md


---

## Dataset Formats Supported (Assignment 2)

### 1. Folder Structure
animals/
├── cat/
│ └── cat_001.jpg
└── dog/
└── dog_001.jpg

### 2. CSV Manifest
```csv
image_path,class_name
images/001.jpg,cat
images/002.jpg,dog
```

### 3. JSON/JSONL Manifest
```json
[{"image_path": "images/001.jpg", "class_name": "cat"}]
```

### 4. NumPy Array (In-Memory)
```python
X = np.array(...)  # (N, H, W, C)
y = np.array([0, 1, ...])
```

---

## Model Selection Guide

| Deployment Scenario | Recommended Model | Reason |
|--------------------|-------------------|--------|
| Mobile phone | MobileNetV3-Small | 6.25 MB, 95.44% acc |
| Edge / IoT | YOLO Classification | 3 MB checkpoint |
| Server (max accuracy) | EfficientNet-B0 | 96.34% test accuracy |
| Fast prototype | Random Forest | Fast train, ~35% acc |
| Interpretability | Decision Tree | Explainable rules |
| Research baseline | ResNet50 | Community standard |

---

## Output Files

### Assignment 2 output (Traditional ML):
benchmark_results/
├── benchmark_summary.csv
├── benchmark_metrics.json
├── model_comparison.png
├── precision_recall_comparison.png
└── confusion_matrix_*.png


### Assignment 3 output (Deep CNN):
cnn_results/
├── checkpoints/best_*.pt # 9 best model checkpoints
├── plots/
│ ├── _curves.png # Training loss/acc per model
│ ├── accuracy_comparison.png
│ ├── f1_comparison.png
│ ├── parameter_comparison.png
│ ├── model_size_comparison.png
│ ├── training_time.png
│ ├── inference_speed.png
│ ├── accuracy_vs_parameters.png
│ └── accuracy_vs_latency.png
└── confusion_matrices/.png # 9 confusion matrices


---

## Testing

```bash
pytest tests/test_benchmark.py -v
```

---

## Assignment Coverage

**Assignment 2 requirements met:**
- Reusable Python package (`pip install Savannah_Shannon_CV_Benchmarking`)
- 4 classical models + 2 neural baselines
- Multiple dataset input formats (folder, CSV, JSON, array)
- Automated metrics + visualizations
- Full test coverage

**Assignment 3 (Part 2) requirements met:**
- 10 CNN architectures (AlexNet, VGG16, ResNet18/50, DenseNet121, MobileNetV3, EfficientNet-B0, GoogLeNet, ConvNeXt-Tiny, YOLO Classification)
- Model factory + unified training loop
- Best checkpoint selection by validation accuracy
- Full metrics: accuracy, precision (macro/weighted), recall, F1 (macro/weighted)
- Confusion matrices for every architecture
- Training curves (loss + accuracy)
- Parameter counts (total + trainable)
- Model file sizes
- Training time (seconds/epoch, total)
- Inference latency + throughput
- GPU memory consumption
- Master benchmark table
- 8 comparison visualizations
- 5 required rankings (see `ANALYSIS.md`)
- Architecture evolution analysis (see `ANALYSIS.md`)

---

## Author

**Savannah Shannon**
- Email: snshannon2002@gmail.com
- Ph.D. Computer Science (AI concentration), Clark Atlanta University

---

## License

MIT License — see LICENSE file.

---

## Acknowledgments

- PyTorch + torchvision for pretrained CNNs
- Ultralytics for YOLO classification
- TensorFlow/Keras for Assignment 2 neural baselines
- Scikit-learn for classical models
- Matplotlib + Seaborn for visualizations
- Google Colab for free T4 GPU compute

---

## Version History

**v2.0.0** (2026-09-18) — Assignment 3 Extension
- Added 10 CNN architectures with pretrained weights
- Full efficiency profiling (latency, throughput, memory)
- 8 comparison visualizations
- Architecture evolution analysis
- YOLO Classification integration

**v1.0.0** (2026-09-08) — Assignment 2 Initial Release
- 4 classical ML + 2 neural models
- 4 dataset formats (folder, CSV, JSON, array)
- Automated metrics & visualizations
- Complete pytest coverage