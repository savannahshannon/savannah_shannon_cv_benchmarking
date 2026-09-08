"""
Classical machine learning models for image classification.
Implements Logistic Regression, Decision Tree, Random Forest, and SVM.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple
import time


class ClassicalModelTrainer:
    # train and evaluate classical ML models

    def __init__(self, random_state: int = 42):
        # initialize classical model trainer
        self.random_state = random_state
        self.models = {}
        self.training_times = {}
        self.inference_times = {}

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        # train all classical models
        results = {}

        # LR
        print("Training Logistic Regression...")
        lr_model, lr_time = self._train_logistic_regression(X_train, y_train)
        results['Logistic Regression'] = {
            'model': lr_model,
            'train_time': lr_time,
            'params': {'C': 1.0, 'max_iter': 1000}
        }
        self.models['Logistic Regression'] = lr_model
        self.training_times['Logistic Regression'] = lr_time

        # DT
        print("Training Decision Tree...")
        dt_model, dt_time = self._train_decision_tree(X_train, y_train)
        results['Decision Tree'] = {
            'model': dt_model,
            'train_time': dt_time,
            'params': {'max_depth': 10}
        }
        self.models['Decision Tree'] = dt_model
        self.training_times['Decision Tree'] = dt_time

        # RF
        print("Training Random Forest...")
        rf_model, rf_time = self._train_random_forest(X_train, y_train)
        results['Random Forest'] = {
            'model': rf_model,
            'train_time': rf_time,
            'params': {'n_estimators': 100, 'max_depth': 10}
        }
        self.models['Random Forest'] = rf_model
        self.training_times['Random Forest'] = rf_time

        # SVM
        print("Training SVM...")
        svm_model, svm_time = self._train_svm(X_train, y_train)
        results['SVM'] = {
            'model': svm_model,
            'train_time': svm_time,
            'params': {'kernel': 'rbf', 'C': 1.0}
        }
        self.models['SVM'] = svm_model
        self.training_times['SVM'] = svm_time

        return results

    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, float]:
        # train LR with feature scaling
        start_time = time.time()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs',
                n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        return pipeline, training_time

    def _train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, float]:
        # train DT with feature scaling
        start_time = time.time()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(
                max_depth=10,
                random_state=self.random_state,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        return pipeline, training_time

    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, float]:
        # train RF with feature scaling
        start_time = time.time()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        return pipeline, training_time

    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, float]:
        # train SVM with feature scaling
        start_time = time.time()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state
            ))
        ])
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        return pipeline, training_time

    def predict_all(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        # get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        return predictions

    def get_inference_times(self, X_test: np.ndarray, n_runs: int = 10) -> Dict[str, float]:
        # measure inference time per image for each model
        inference_times = {}

        for name, model in self.models.items():
            times = []
            for _ in range(n_runs):
                start = time.time()
                model.predict(X_test)
                elapsed = time.time() - start
                times.append(elapsed)

            # avg time per image in ms
            avg_time = np.mean(times) / len(X_test) * 1000
            inference_times[name] = avg_time

        self.inference_times = inference_times
        return inference_times
