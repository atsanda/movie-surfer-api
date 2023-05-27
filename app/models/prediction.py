import numpy as np

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: list[int]


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    n: int
    seen_movies: list[int]
