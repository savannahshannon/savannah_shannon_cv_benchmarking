"""
Visualization module for benchmark results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class BenchmarkVisualizer:
    # visualize benchmark results and model comparisons

    def __init__(self, output_dir: str = './results', figsize: tuple = (12, 8)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize

    def plot_class_distribution(self, y_data: np.ndarray, class_names: List[str],
                                title: str = "Class Distribution") -> None:
        # plot histogram of class distribution
        plt.figure(figsize=self.figsize)
        unique, counts = np.unique(y_data, return_counts=True)
        class_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in unique]

        plt.bar(class_labels, counts, color='steelblue', edgecolor='black')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / 'class_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        # plot 4-panel model comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Comparison", fontsize=16, fontweight='bold')

        models = results_df['Model']
        colors = sns.color_palette("husl", len(models))

        # accuracy
        axes[0, 0].bar(models, results_df['Accuracy'], color=colors, edgecolor='black')
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)

        # macro f1
        axes[0, 1].bar(models, results_df['Macro F1'], color=colors, edgecolor='black')
        axes[0, 1].set_title('Macro F1 Score', fontweight='bold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)

        # training time
        axes[1, 0].bar(models, results_df['Training Time (s)'], color=colors, edgecolor='black')
        axes[1, 0].set_title('Training Time', fontweight='bold')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # inference time
        axes[1, 1].bar(models, results_df['Inference Time (ms/image)'], color=colors, edgecolor='black')
        axes[1, 1].set_title('Inference Time per Image', fontweight='bold')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, model_name: str,
                            class_names: List[str]) -> None:
        # plot confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        # sanitize filename
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        output_path = self.output_dir / f'confusion_matrix_{safe_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def plot_precision_recall_comparison(self, results_df: pd.DataFrame) -> None:
        # plot precision vs recall comparison
        plt.figure(figsize=self.figsize)
        models = results_df['Model']
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, results_df['Macro Precision'], width, label='Precision', edgecolor='black')
        plt.bar(x + width/2, results_df['Macro Recall'], width, label='Recall', edgecolor='black')

        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision vs Recall Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.ylim([0, 1])
        plt.tight_layout()

        output_path = self.output_dir / 'precision_recall_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def plot_split_distribution(self, train_size: int, test_size: int) -> None:
        # plot train/test split distribution
        plt.figure(figsize=(10, 6))
        sizes = [train_size, test_size]
        labels = ['Training Set', 'Test Set']
        colors = ['#66c2a5', '#fc8d62']
        total = train_size + test_size

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        plt.title(f"Train/Test Split Distribution\n(Total: {total} samples)",
                fontsize=14, fontweight='bold')

        output_path = self.output_dir / 'split_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def save_all_plots(self, results_df: pd.DataFrame, confusion_matrices: Dict,
                    y_full: np.ndarray, class_names: List[str],
                    train_size: int, test_size: int) -> None:
        print("Generating visualizations...")

        self.plot_class_distribution(y_full, class_names)
        self.plot_model_comparison(results_df)
        self.plot_precision_recall_comparison(results_df)
        self.plot_split_distribution(train_size, test_size)

        for model_name, cm in confusion_matrices.items():
            self.plot_confusion_matrix(cm, model_name, class_names)

        print(f"\nAll plots saved to: {self.output_dir.absolute()}")
