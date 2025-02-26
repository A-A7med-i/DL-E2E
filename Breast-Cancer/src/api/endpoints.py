import io
import yaml
import numpy as np
from PIL import Image
from typing import List
import tensorflow as tf
from functools import lru_cache
from fastapi import APIRouter, File, UploadFile
from schemas import PredictionResponse, BatchPredictionResponse

PREDICTION_MAPPING = {0: "benign", 1: "malignant"}

router = APIRouter()


@lru_cache(maxsize=1)
def load_config():
    path = "/media/ahmed/Data/DL-E2E/Breast-Cancer/config/config.yaml"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
        return config


@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(load_config()["model"]["path"])


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    return np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)


@router.get("/")
async def hello():
    return {"message": "Hello world"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_content = await file.read()
    processed_image = preprocess_image(image_content)

    prediction = model.predict(processed_image)

    prediction_value = round(float(prediction[0][0]))
    prediction_str = PREDICTION_MAPPING[prediction_value]

    return PredictionResponse(filename=file.filename, prediction=prediction_str)


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    predictions = []

    for file in files:
        image_content = await file.read()
        processed_image = preprocess_image(image_content)

        prediction = model.predict(processed_image)

        prediction_value = round(float(prediction[0][0]))
        prediction_str = PREDICTION_MAPPING[prediction_value]

        predictions.append(
            PredictionResponse(filename=file.filename, prediction=prediction_str)
        )

    return BatchPredictionResponse(predictions=predictions)


model = load_model()
