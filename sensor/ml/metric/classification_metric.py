import os, sys
from sensor.logger import logging
from sensor.exception import SensorException
from sklearn.metrics import f1_score, precision_score, recall_score
from sensor.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true, y_pred):
    """
        :param y_true:
        :param y_pred:
        :return: classification metric
    """
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)

        #creating artifact
        classification_metric = ClassificationMetricArtifact(
            f1_score = model_f1_score,
            precision_score = model_precision_score,
            recall_score = model_recall_score
        )

        return classification_metric

    except Exception as e:
        raise SensorException(e, sys)

