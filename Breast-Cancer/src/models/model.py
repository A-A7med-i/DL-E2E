from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class ImageClassifier:
    """
    A deep learning classifier for image classification tasks using ResNet50 backbone.

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape of images (height, width, channels)
        backbone (tf.keras.Model): Feature extraction backbone model
        model (tf.keras.Model): Complete classification model
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> None:
        """
        Initialize the classifier with specified input shape.

        Args:
            input_shape: Tuple specifying input dimensions (height, width, channels)
        """
        self.input_shape = input_shape
        self.backbone = self._create_backbone()
        self.model = self._build_architecture()

    def _create_backbone(self) -> tf.keras.Model:
        """
        Creates and configures the ResNet50 backbone.

        Returns:
            Configured ResNet50 model for feature extraction
        """
        backbone = ResNet50(include_top=False, input_shape=self.input_shape)
        backbone.trainable = False
        return backbone

    def _build_architecture(self) -> tf.keras.Model:
        """
        Builds the complete model architecture.

        Returns:
            Compiled Keras model ready for training
        """
        inputs = Input(shape=self.input_shape)

        x = self.backbone(inputs)

        x = GlobalAveragePooling2D()(x)

        x = Dense(64)(x)
        x = Activation("relu")(x)

        outputs = Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs, outputs)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
    ) -> tf.keras.callbacks.History:
        """
        Trains the model on provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs

        Returns:
            Training history
        """
        callbacks = [EarlyStopping(monitor="loss", patience=5)]

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall"],
        )

        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
        )

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates model performance on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.model.evaluate(X_test, y_test)
        return {
            "loss": results[0],
            "accuracy": results[1],
            "precision": results[2],
            "recall": results[3],
        }

    def plot_training_history(self, history: tf.keras.callbacks.History) -> None:
        """
        Visualizes training metrics history with accuracy and loss curves.

        Args:
            history: Training history containing metrics from model.fit()
        """
        plt.figure(figsize=(14, 6))

        # Plot accuracy
        plt.plot(
            history.history["accuracy"], label="Accuracy", color="#1f77b4", linewidth=2
        )

        # Plot loss
        plt.plot(
            history.history["loss"],
            label="Loss",
            color="#d62728",
            linestyle="--",
            linewidth=2,
        )

        plt.title("Training Metrics", fontsize=14)
        plt.ylabel("Metrics Value", fontsize=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for input data.

        Args:
            X: Input data for prediction

        Returns:
            Array of predicted probabilities
        """
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Saves model to disk.

        Args:
            filepath: Path where model will be saved
        """
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Loads model from disk.

        Args:
            filepath: Path to saved model
        """
        self.model = tf.keras.models.load_model(filepath)

    def get_model_summary(self) -> str:
        """
        Returns model architecture summary.

        Returns:
            String representation of model architecture
        """
        return self.model.summary()
