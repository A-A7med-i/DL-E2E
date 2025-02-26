from typing import List
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """
    Schema for single image prediction response.

    Attributes:
        filename (str): Name of the uploaded image file
        prediction (str): Model prediction result ('benign' or 'malignant')
    """

    filename: str
    prediction: str


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch image prediction response.

    Attributes:
        predictions (List[PredictionResponse]): List of prediction results for multiple images
    """

    predictions: List[PredictionResponse]
