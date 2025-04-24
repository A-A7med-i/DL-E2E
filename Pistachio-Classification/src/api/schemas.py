from pydantic import BaseModel
from typing import Literal


class PredictionResponse(BaseModel):
    """
    Response model for image classification predictions.

    Attributes:
        FileName (str): The name of the uploaded file that was classified.
        Prediction (str): The prediction result, either "Kirmizi_Pistachio" or "Siirt_Pistachio".
    """

    FileName: str
    probability: float
    PredictionClass: Literal["Kirmizi_Pistachio", "Siirt_Pistachio"]
