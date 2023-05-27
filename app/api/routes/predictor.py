import json

import joblib
from fastapi import APIRouter, HTTPException

from core.config import INPUT_EXAMPLE
from services.predict import MachineLearningModelHandlerScore as model
from models.prediction import (
    HealthResponse,
    MachineLearningResponse,
    MachineLearningDataInput,
)

router = APIRouter()


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(data_point):
    return model.predict(dict(data_point), method="predict")


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: MachineLearningDataInput):

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        prediction = get_prediction(data_input)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return MachineLearningResponse(prediction=prediction)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
)
async def health():
    is_health = False
    try:
        test_input = MachineLearningDataInput(
            **json.loads(open(INPUT_EXAMPLE, "r").read())
        )
        get_prediction(test_input)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")
