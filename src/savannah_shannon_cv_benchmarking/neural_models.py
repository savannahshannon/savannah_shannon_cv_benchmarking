"""
Neural network models for image classification
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, Tuple

# import TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class NeuralNetworkTrainer:
    # train and evaluate neural network models

    def __init__(self, random_state: int = 42, epochs: int = 20, batch_size: int = 32):
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = {}
        self.training_times = {}
        self.model_configs = {}

    def train_all_models(self,
        X_train: np.ndarray,
        y_train: np.ndarray) -> Dict:
        """Train all 2 NN models."""
        results = {}

        num_classes = len(np.unique(y_train))

        # 1. fully connected MLP
        print("Training Fully Connected Neural Network")
        results['Neural Network'] = self._train_mlp(
            X_train, y_train, num_classes
        )

        # 2. CNN (or fallback to Gradient Boosting)
        print("Training Convolutional Neural Network")
        if TENSORFLOW_AVAILABLE:
            results['Simple CNN'] = self._train_cnn_tf(
                X_train, y_train, num_classes
            )
        else:
            print("TensorFlow not available, using Gradient Boosting fallback")
            results['Simple CNN'] = self._train_cnn_sklearn(
                X_train, y_train, num_classes
            )
        return results

    def _train_mlp(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int) -> Dict:
        # train fully connected MLP (Multi-Layer Perceptron)
        start_time = time.time()

        # flatten images for MLP
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        if TENSORFLOW_AVAILABLE:
            model = self._build_mlp_keras(X_train_flat.shape, num_classes)

            # train
            history = model.fit(
                X_train_flat, tf.keras.utils.to_categorical(y_train, num_classes),
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0
            )

            train_time = time.time() - start_time
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation='relu',
                    solver='adam',
                    max_iter=self.epochs,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ])

            model.fit(X_train_flat, y_train)
            train_time = time.time() - start_time

        self.models['Neural Network'] = model
        self.training_times['Neural Network'] = train_time
        self.model_configs['Neural Network'] = {
            'architecture': 'MLP (128 -> 64)',
            'activation': 'ReLU',
            'optimizer': 'Adam',
            'loss': 'Cross-Entropy',
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'framework': 'TensorFlow' if TENSORFLOW_AVAILABLE else 'scikit-learn'
        }

        return {
            'model': model,
            'train_time': train_time
        }

    def _build_mlp_keras(self, input_shape: Tuple, num_classes: int):
        """Build keras MLP model."""
        model = keras.Sequential([
            layers.Input(shape=(int(input_shape[1]),)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _train_cnn_tf(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int) -> Dict:
        """Train CNN using TensorFlow/Keras."""
        start_time = time.time()

        # normalize images to [0,1]
        X_train_norm = X_train.astype(np.float32)

        # build model
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # train
        history = model.fit(
            X_train_norm, tf.keras.utils.to_categorical(y_train, num_classes),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )

        train_time = time.time() - start_time

        self.models['Simple CNN'] = model
        self.training_times['Simple CNN'] = train_time
        self.model_configs['Simple CNN'] = {
            'architecture': 'CNN (Conv32→Conv64→Dense128)',
            'activation': 'ReLU',
            'optimizer': 'Adam',
            'loss': 'Cross-Entropy',
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'framework': 'TensorFlow'
        }

        return {
            'model': model,
            'train_time': train_time
        }

    def _train_cnn_sklearn(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            num_classes: int) -> Dict:
        # fallback CNN using gradient boosting (when TensorFlow is not available)
        start_time = time.time()

        # flatten images
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('gbc', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ))
        ])

        model.fit(X_train_flat, y_train)
        train_time = time.time() - start_time

        self.models['Simple CNN'] = model
        self.training_times['Simple CNN'] = train_time
        self.model_configs['Simple CNN'] = {
            'architecture': 'Gradient Boosting (fallback)',
            'optimizer': 'Gradient Descent',
            'loss': 'Deviance',
            'n_estimators': 100,
            'framework': 'scikit-learn'
        }

        return {
            'model': model,
            'train_time': train_time
        }

    def predict_all(self, X_test: np.ndarray) -> Dict:
        predictions = {}

        for model_name, model in self.models.items():
            X_test_input = X_test

            # flatten if needed for MLP/sklearn models
            if model_name == 'Neural Network':
                X_test_input = X_test.reshape(X_test.shape[0], -1)
            elif model_name == 'Simple CNN' and not isinstance(model, keras.Model):
                X_test_input = X_test.reshape(X_test.shape[0], -1)

            if TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
                predictions[model_name] = model.predict(X_test_input, verbose=0).argmax(axis=1)
            else:
                predictions[model_name] = model.predict(X_test_input)

        return predictions

    def get_inference_times(self, X_test: np.ndarray) -> Dict:
        # measure average inference time per sample for each model
        times = {}
        num_runs = 5

        for model_name, model in self.models.items():
            model_times = []

            for _ in range(num_runs):
                X_test_input = X_test

                # Flatten if needed for MLP/sklearn models
                if model_name == 'Neural Network':
                    X_test_input = X_test.reshape(X_test.shape[0], -1)
                elif model_name == 'Simple CNN' and not isinstance(model, keras.Model):
                    X_test_input = X_test.reshape(X_test.shape[0], -1)

                start = time.time()
                if TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
                    _ = model.predict(X_test_input, verbose=0)
                else:
                    _ = model.predict(X_test_input)
                model_times.append(time.time() - start)

            # avg time per sample in ms
            avg_time_per_sample = (np.mean(model_times) / len(X_test)) * 1000
            times[model_name] = avg_time_per_sample

        return times
