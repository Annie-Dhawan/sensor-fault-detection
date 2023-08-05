import shutil
import os, sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.config_entity import ModelPusherConfig
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
from sensor.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact, ModelEvaluationArtifact


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig, model_eval_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            # using this to know the file path of the model
            self.model_eval_artifact = model_eval_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self):
        try:
            trained_model_path = self.model_eval_artifact.trained_model_file_path

            # Creating model pusher dir to save model, this is for training pipeline
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            #copying the file
            shutil.copy(src=trained_model_path, dst=model_file_path)

            # saved model dir, this model is for the production purpose
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            # prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path,
                                                        model_file_path=model_file_path)
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys)
