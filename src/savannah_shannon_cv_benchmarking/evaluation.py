"""
Model evaluation module.

Computes comprehensive evaluation metrics for all models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, List


class ModelEvaluator:
    """Evaluate model predictions and generate comprehensive reports."""

    def __init__(self):
        # initialize evaluator
        self.results = None
        self.confusion_matrices = {}
        self.classification_reports = {}

    def evaluate_all_predictions(self,
            predictions: Dict[str, np.ndarray],
            y_test: np.ndarray,
            training_times: Dict[str, float],
            inference_times: Dict[str, float],
            class_names: List[str]) -> pd.DataFrame:
        # evaluate all model predictions and generate metrics
        results = []

        for model_name, y_pred in predictions.items():
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Macro Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Macro Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'Macro F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'Weighted F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'Training Time (s)': training_times.get(model_name, 0),
                'Inference Time (ms/image)': inference_times.get(model_name, 0),
            }
            results.append(metrics)

            # store confusion matrix and classification report
            self.confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
            self.classification_reports[model_name] = classification_report(
                y_test, y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )

        # create DataFrame and sort by Macro F1 (descending) then Inference Time (ascending)
        df = pd.DataFrame(results)
        df = df.sort_values(by=['Macro F1', 'Inference Time (ms/image)'],
                ascending=[False, True]).reset_index(drop=True)

        self.results = df
        return df

    def get_confusion_matrix(self, model_name: str) -> np.ndarray:
        if model_name not in self.confusion_matrices:
            raise ValueError(f"No confusion matrix found for {model_name}")
        return self.confusion_matrices[model_name]

    def get_classification_report(self, model_name: str) -> Dict:
        if model_name not in self.classification_reports:
            raise ValueError(f"No classification report found for {model_name}")
        return self.classification_reports[model_name]

    def get_best_model(self) -> Tuple[str, Dict]:
        if self.results is None:
            raise ValueError("No evaluation results available")

        best_idx = 0
        best_model = self.results.iloc[best_idx]

        return best_model['Model'], best_model.to_dict()

    def get_summary_stats(self) -> Dict[str, float]:
        if self.results is None:
            raise ValueError("No evaluation results available")

        return {
            'mean_accuracy': self.results['Accuracy'].mean(),
            'std_accuracy': self.results['Accuracy'].std(),
            'mean_macro_f1': self.results['Macro F1'].mean(),
            'std_macro_f1': self.results['Macro F1'].std(),
            'mean_training_time': self.results['Training Time (s)'].mean(),
            'mean_inference_time': self.results['Inference Time (ms/image)'].mean(),
        }

    def print_report(self, verbose: bool = True):
        if self.results is None:
            print("No evaluation results available")
            return

        print("MODEL EVALUATION REPORT")
        print(self.results.to_string(index=False))

        if verbose:
            print("\nSUMMARY STATISTICS:")
            stats = self.get_summary_stats()
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")

            print("\nBEST MODEL:")
            best_model, metrics = self.get_best_model()
            print(f"Model: {best_model}")
            print(f"Macro F1: {metrics['Macro F1']:.4f}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
