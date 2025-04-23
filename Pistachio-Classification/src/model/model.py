import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple


class BinaryClassificationModel:
    """
    A binary classification model based on ResNet50 architecture.

    This model uses transfer learning with a pre-trained ResNet50 as the base model
    and adds custom classification layers on top for binary classification tasks.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (300, 300, 3),
        learning_rate: float = 0.001,
    ):
        """
        Initialize the binary classification model.

        Args:
            input_shape: Tuple specifying the input image dimensions (height, width, channels)
            learning_rate: Initial learning rate for the Adam optimizer
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.base_model = ResNet50(
            include_top=False, input_shape=self.input_shape, weights="imagenet"
        )
        self.base_model.trainable = False
        self.model = self.build_architecture()

    def build_architecture(self) -> tf.keras.Model:
        """
        Build the model architecture with ResNet50 as the base.

        Returns:
            A compiled Keras model ready for training
        """
        input_layer = Input(shape=self.input_shape)

        x = self.base_model(input_layer)

        x = GlobalAveragePooling2D()(x)

        x = Dense(256, activation="relu")(x)

        output_layer = Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(
            inputs=input_layer, outputs=output_layer, name="binary_classifier"
        )

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        epochs: int = 20,
        validation_split: float = 0.1,
    ) -> tf.keras.callbacks.History:
        """
        Train the model with the provided data.

        Args:
            x_train: Training images as numpy array
            y_train: Training labels as numpy array
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            validation_split: Fraction of the training data to be used as validation data

        Returns:
            A History object containing training metrics
        """
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
        )

        return history

    def evaluate_model(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x_test: Test images as numpy array
            y_test: Test labels as numpy array

        Returns:
            Dictionary with evaluation metrics (loss, accuracy, precision, recall)
        """
        results = self.model.evaluate(x_test, y_test, verbose=1)

        metrics_dict = {
            "loss": round(results[0], 3),
            "accuracy": round(results[1], 3),
            "precision": round(results[2], 3),
            "recall": round(results[3], 3),
        }

        return metrics_dict

    def plot_training_history(
        self, history: tf.keras.callbacks.History, figsize: Tuple[int, int] = (15, 6)
    ) -> None:
        """
        Plot training and validation metrics from the training history.

        Args:
            history: History object returned by model.fit()
            figsize: Figure size for the plots (width, height)
        """
        history_dict = history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(history_dict["accuracy"], label="Train", color="#1f77b4", linewidth=2)
        ax1.plot(
            history_dict["val_accuracy"],
            label="Validation",
            color="#ff7f0e",
            linestyle="--",
        )
        ax1.set_title("Model Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(history_dict["loss"], label="Train", color="#2ca02c", linewidth=2)
        ax2.plot(
            history_dict["val_loss"],
            label="Validation",
            color="#d62728",
            linestyle="--",
        )
        ax2.set_title("Loss")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Generates predictions for input data.

        Args:
            X: Input data for prediction

        Returns:
            Array of predicted probabilities
        """
        probabilities = self.model.predict(x)

        return probabilities

    def get_model_summary(self) -> str:
        """
        Returns model architecture summary.

        Returns:
            String representation of model architecture
        """
        return self.model.summary()

    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path where the model should be saved
        """
        self.model.save(filepath)
        print(f"Model successfully saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model successfully loaded from {filepath}")
