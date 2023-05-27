import os

from loguru import logger
import mlflow

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_NAME, MLFLOW_TRACKING_URI


class MachineLearningModelHandlerScore(object):
    model = None

    @classmethod
    def predict(cls, input: dict, method="predict"):
        clf = cls.get_model()
        if hasattr(clf, method):
            return getattr(clf, method)(input)
        raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = cls.load()
        return cls.model

    @staticmethod
    def load():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
        latest = versions[-1].version
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest}")
        return model
