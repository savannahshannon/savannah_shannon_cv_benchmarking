# generate all 8 required comparison plots from the master CSV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('cnn_results/plots', exist_ok=True)
df = pd.read_csv('combined_ml_cnn_benchmark_results.csv')

# family colors for consistent visual grouping
def get_family(model):
    m = model.lower()
    if m in ['logistic_regression', 'decision_tree', 'random_forest', 'svm']: return 'Traditional ML'
    if m in ['mlp', 'simple_cnn']: return 'Baseline'
    if m == 'alexnet': return 'AlexNet'
    if 'vgg' in m: return 'VGG'
    if m == 'googlenet': return 'Inception'
    if 'resnet' in m: return 'ResNet'
    if 'densenet' in m: return 'DenseNet'
    if 'mobilenet' in m: return 'MobileNet'
    if 'efficientnet' in m: return 'EfficientNet'
    if 'convnext' in m: return 'ConvNeXt'
    if 'yolo' in m: return 'YOLO'
    return 'Other'

if 'Family' not in df.columns:
    df['Family'] = df['Model'].apply(get_family)

# convert params to millions for cleaner axes
df['Params_M'] = df['Total_Parameters'] / 1e6

FAMILY_COLORS = {
    'Traditional ML': '#8B8B8B', 'Baseline': '#B8B8B8',
    'AlexNet': '#E76F51', 'VGG': '#F4A261', 'Inception': '#E9C46A',
    'ResNet': '#2A9D8F', 'DenseNet': '#264653', 'MobileNet': '#457B9D',
    'EfficientNet': '#1D3557', 'ConvNeXt': '#6A0572', 'YOLO': '#A4133C'
}

def plot_bar(x_col, title, xlabel, filename, sort=True):
    d = df.dropna(subset=[x_col]).sort_values(x_col, ascending=True) if sort else df
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = [FAMILY_COLORS.get(f, '#888') for f in d['Family']]
    ax.barh(d['Model'], d[x_col], color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    for i, v in enumerate(d[x_col]):
        ax.text(v, i, f' {v:.3f}' if v < 10 else f' {v:.1f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'cnn_results/plots/{filename}', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'✓ {filename}')

def plot_scatter(x_col, y_col, xlabel, ylabel, title, filename, log_x=False):
    d = df.dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=(11, 7))
    for family in d['Family'].unique():
        sub = d[d['Family'] == family]
        ax.scatter(sub[x_col], sub[y_col], s=180, alpha=0.75,
                color=FAMILY_COLORS.get(family, '#888'), label=family, edgecolors='white', linewidth=1.5)
    for _, row in d.iterrows():
        ax.annotate(row['Model'], (row[x_col], row[y_col]),
                    xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if log_x: ax.set_xscale('log')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cnn_results/plots/{filename}', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'✓ {filename}')

# 8 required plots
plot_bar('Test_Accuracy', 'Model vs. Test Accuracy', 'Test Accuracy', 'accuracy_comparison.png')
plot_bar('F1_Macro', 'Model vs. F1 Score (Macro)', 'F1 Score (Macro)', 'f1_comparison.png')
plot_bar('Params_M', 'Model vs. Number of Parameters', 'Parameters (Millions)', 'parameter_comparison.png')
plot_bar('Model_Size_MB', 'Model vs. Model Size', 'Model Size (MB)', 'model_size_comparison.png')
plot_bar('Training_Time_sec', 'Model vs. Training Time', 'Training Time (seconds)', 'training_time.png')
plot_bar('Throughput_img_per_sec', 'Model vs. Inference Throughput', 'Images per Second', 'inference_speed.png')
plot_scatter('Params_M', 'Test_Accuracy', 'Parameters (Millions, log)', 'Test Accuracy',
            'Accuracy vs. Number of Parameters', 'accuracy_vs_parameters.png', log_x=True)
plot_scatter('Inference_Latency_ms', 'Test_Accuracy', 'Inference Latency (ms)', 'Test Accuracy',
            'Accuracy vs. Inference Latency', 'accuracy_vs_latency.png')

print(f'\n✓ All 8 plots saved to cnn_results/plots/')