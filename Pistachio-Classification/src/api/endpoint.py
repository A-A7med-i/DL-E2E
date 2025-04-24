from flask import Blueprint, request, jsonify
from .schemas import PredictionResponse
from utils.helper import load_yaml
from functools import lru_cache
from constants import *
import tensorflow as tf
from PIL import Image
import numpy as np
import io


router = Blueprint("api", __name__)


try:
    config = load_yaml(CONFIG_PATH)
    model_path = config["model"]["path"]

except Exception as e:
    print(f"Failed to load configuration: {str(e)}")


@lru_cache(maxsize=1)
def load_model(path):
    """
    Load the TensorFlow model from the specified path.

    Args:
        path (str): Path to the saved model

    Returns:
        tf.keras.Model: The loaded model
    """
    try:
        model = tf.keras.models.load_model(path)
        return model

    except Exception as e:
        print(f"Failed to load model: {str(e)}")


try:
    model = load_model(model_path)
except Exception as e:
    print(f"Model initialization failed: {str(e)}")


def preprocess_image(image_bytes):
    """
    Preprocess the image for model prediction.

    Args:
        image_bytes (bytes): Raw image bytes

    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        image = image.resize((300, 300))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")


@router.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint for image classification.

    Accepts an image file and returns the classification result.

    Returns:
        JSON: A prediction response with filename and class prediction
    """
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_content = file.read()

        processed_image = preprocess_image(image_content)

        probability = round(model.predict(processed_image)[0][0], 4)

        prediction_class = (
            "Siirt_Pistachio" if round(probability) else "Kirmizi_Pistachio"
        )

        response = PredictionResponse(
            FileName=file.filename,
            probability=probability,
            PredictionClass=prediction_class,
        )

        return jsonify(response.model_dump())

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
