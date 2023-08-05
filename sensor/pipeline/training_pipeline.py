import os, sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.components.model_pusher import ModelPusher
from sensor.components.model_trainer import ModelTrainer
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.data_transformation import DataTransformation
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig


class TrainPipeline:

    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        #self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)

    def start_data_ingestion(self):

        """
        :return: This function will return train and test file path
        """
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config) #this is the object for dataingestion class
            data_ingestion_artifact = data_ingestion.inititate_data_ingestion()
            logging.info(f"Data ingestion is completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data Validation")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation is completed and artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_data_transformation(self, data_validation_artifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Transformation")
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.inititate_data_transformation()
            logging.info(f"Data Transformation is completed and artifact: {data_validation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Model Training")
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training is completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_model_evaluation(self,data_validation_artifact, model_trainer_artifact):
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Model Evaluation")
            model_eval = ModelEvaluation(model_evaluation_config, data_validation_artifact, model_trainer_artifact)
            model_evaluation_artifact = model_eval.initiate_model_evaluation()
            logging.info(f"Model Evaluation is completed and artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_model_pusher(self, model_eval_artifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Model Pusher")
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            modal_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model Pusher is completed and artifact: {modal_pusher_artifact}")
            return modal_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact, model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            TrainPipeline.is_pipeline_running = False
        except Exception as e:
            TrainPipeline.is_pipeline_running = False
            raise SensorException(e, sys)
