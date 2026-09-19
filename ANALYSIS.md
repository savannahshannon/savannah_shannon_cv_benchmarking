# CV Benchmarking Analysis — Rankings & Architecture Evolution

## 1. Architecture Evolution

**AlexNet (2012)** — Broke through the ImageNet challenge with ReLU activations, Dropout regularization, and GPU-based training. First proof that deep CNNs could dominate hand-crafted features. Its 57M parameters, dominated by fully-connected layers (75.94% CIFAR-10 in my benchmark), reflect a design predating parameter-efficient conventions.

**VGG16 (2014)** — Simplified the design philosophy: just stack 3×3 convolutions. Proved depth matters more than kernel size, but at heavy cost (134M parameters, 537 MB checkpoint). In my run it underperformed (77.07%) — evidence that raw depth without residual connections struggles to converge cleanly in only 20 epochs.

**GoogLeNet / Inception (2014)** — Introduced Inception modules that apply multiple receptive field sizes in parallel, using 1×1 convolutions for dimensionality reduction. Achieves 95.39% accuracy with only 12M parameters — 11× smaller than VGG16.

**ResNet (2015)** — Solved the vanishing gradient problem with skip connections. My results: ResNet18 → 93.98%, ResNet50 → 94.81%. Depth increased 2.8× but accuracy improved only 0.83%, suggesting diminishing returns at scale on CIFAR-10.

**DenseNet121 (2017)** — Every layer receives inputs from all previous layers. Achieves 95.27% with just 7M parameters — 3× more parameter-efficient than ResNet50 for similar accuracy.

**MobileNetV3-Small (2019)** — Depthwise separable convolutions plus neural architecture search. My benchmark's efficiency king: **95.44% accuracy with only 1.53M parameters and 6.25 MB checkpoint** — 89× smaller than VGG16 with 18 percentage points higher accuracy.

**EfficientNet-B0 (2019)** — Compound scaling of depth, width, and resolution. My best performer: **96.34% test accuracy** with 4M parameters. Best accuracy-to-size ratio in the benchmark.

**ConvNeXt-Tiny (2022)** — Modernized ResNet with Transformer-inspired design (7×7 depthwise conv, LayerNorm, GELU). 95.99% accuracy with 28M parameters — proves classical convnets still competitive with ViTs.

**YOLO Classification (v8n-cls)** — YOLO's backbone repurposed for pure classification. In my run: 95.1% accuracy with just 1.45M parameters and 3 MB checkpoint. Extreme efficiency at cost of ~1 pp accuracy vs. EfficientNet.

## 2. Historical Timeline
Traditional ML (LR/DT/RF/SVM) → Hand-crafted features
↓ Problem: cannot learn hierarchical features from raw pixels
MLP / Simple CNN → Learned features, but shallow
↓ Problem: shallow networks miss abstract concepts
AlexNet (2012) → Deep learning breakthrough
↓ Problem: depth causes vanishing gradients
VGG (2014) → Standardized deep architecture
↓ Problem: too many parameters, hard to train
GoogLeNet (2014) → Multi-scale via Inception
↓ Problem: 100+ layers won't converge
ResNet (2015) → Skip connections
↓ Problem: parameters grow with depth
DenseNet (2017) → Dense feature reuse
↓ Problem: not efficient for edge devices
MobileNet (2017-2019) → Depthwise separable conv
↓ Problem: manual scaling is suboptimal
EfficientNet (2019) → Principled compound scaling
↓ Problem: convnets losing to Transformers on large data
ConvNeXt (2022) → Modernized convnet
↓ Problem: need fast unified detection/classification
YOLO Classification (2023+) → Compact, high-throughput

## 3. Rankings

### Ranking A — Highest Accuracy

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | EfficientNet-B0 | 96.34% |
| 2 | ConvNeXt-Tiny | 95.99% |
| 3 | MobileNetV3-Small | 95.44% |
| 4 | GoogLeNet | 95.39% |
| 5 | DenseNet121 | 95.27% |
| 6 | YOLO Classification | 95.10% |
| 7 | ResNet50 | 94.81% |
| 8 | ResNet18 | 93.98% |
| 9 | VGG16 | 77.07% |
| 10 | AlexNet | 75.94% |

### Ranking B — Fastest Inference (Images/sec)

| Rank | Model | Throughput (img/s) | Latency (ms) |
|------|-------|-------------------|--------------|
| 1 | AlexNet | 476.36 | 2.10 |
| 2 | ResNet18 | 375.57 | 2.66 |
| 3 | MobileNetV3-Small | 149.36 | 6.70 |
| 4 | ConvNeXt-Tiny | 147.41 | 6.78 |
| 5 | ResNet50 | 145.39 | 6.88 |
| 6 | GoogLeNet | 123.49 | 8.10 |
| 7 | VGG16 | 102.68 | 9.74 |
| 8 | EfficientNet-B0 | 95.04 | 10.52 |
| 9 | DenseNet121 | 56.09 | 17.83 |

### Ranking C — Smallest Model (Checkpoint Size)

| Rank | Model | Size (MB) |
|------|-------|-----------|
| 1 | YOLO Classification | 3.0 |
| 2 | MobileNetV3-Small | 6.25 |
| 3 | EfficientNet-B0 | 16.38 |
| 4 | DenseNet121 | 28.47 |
| 5 | ResNet18 | 44.81 |
| 6 | GoogLeNet | 48.15 |
| 7 | ResNet50 | 94.43 |
| 8 | ConvNeXt-Tiny | 111.38 |
| 9 | AlexNet | 228.19 |
| 10 | VGG16 | 537.22 |

### Ranking D — Accuracy per Million Parameters

| Rank | Model | Params (M) | Acc/M Params |
|------|-------|------------|--------------|
| 1 | YOLO Classification | 1.45 | 65.59 |
| 2 | MobileNetV3-Small | 1.53 | 62.38 |
| 3 | EfficientNet-B0 | 4.02 | 23.97 |
| 4 | DenseNet121 | 6.96 | 13.69 |
| 5 | ResNet18 | 11.18 | 8.41 |
| 6 | GoogLeNet | 11.99 | 7.96 |
| 7 | ResNet50 | 23.53 | 4.03 |
| 8 | ConvNeXt-Tiny | 27.83 | 3.45 |
| 9 | AlexNet | 57.04 | 1.33 |
| 10 | VGG16 | 134.30 | 0.57 |

⚠️ *Note: This is a simple ratio, not a universal quality metric.*

### Ranking E — Overall Recommendation

**Scoring: 40% Accuracy + 20% F1 + 15% Speed + 15% Size + 10% Memory (each normalized 0-1)**

| Rank | Model | Composite Score | Best For |
|------|-------|-----------------|----------|
| 🥇 1 | EfficientNet-B0 | 0.92 | Server-side accuracy |
| 🥈 2 | MobileNetV3-Small | 0.91 | Mobile / edge |
| 🥉 3 | YOLO Classification | 0.87 | Real-time embedded |
| 4 | ConvNeXt-Tiny | 0.80 | Modern accurate |
| 5 | GoogLeNet | 0.76 | Balanced legacy |
| 6 | DenseNet121 | 0.72 | Feature-rich |
| 7 | ResNet50 | 0.65 | Deep baseline |
| 8 | ResNet18 | 0.63 | Fast baseline |
| 9 | VGG16 | 0.30 | Historical |
| 10 | AlexNet | 0.28 | Historical |

## 4. Deployment Recommendations

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Mobile phone** | MobileNetV3-Small | 6.25 MB, 95.44% acc |
| **UAV / drone** | MobileNetV3-Small or YOLO | Sub-10 MB checkpoints |
| **Edge IoT** | YOLO Classification | 3 MB, fast inference |
| **Server (max accuracy)** | EfficientNet-B0 | 96.34% test accuracy |
| **Real-time video** | MobileNetV3-Small | Best speed/accuracy balance |
| **Research baseline** | ResNet50 | Community standard |

## 5. Key Findings

1. **Modern architectures beat parameter count.** MobileNetV3-Small (1.53M params) outperforms ResNet50 (23.5M params) — 15× fewer parameters, higher accuracy.

2. **Old architectures underperform on modern benchmarks.** AlexNet and VGG16 (2012-2014) plateau at 75-77% while every architecture from 2014+ hits 93%+.

3. **Persistent errors: Cat ↔ Dog confusion.** All models struggle with visually similar animal classes. EfficientNet-B0 has fewest such errors (44 vs. 211 for AlexNet).

4. **Trade-offs are real:** DenseNet121 has highest inference latency (17.83 ms) despite modest size. VGG16 is huge but not much faster than modern nets.

5. **Compound scaling wins.** EfficientNet-B0 achieves best accuracy with just 4M parameters — validates the principled scaling approach.