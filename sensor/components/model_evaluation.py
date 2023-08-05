import os, sys
import numpy as np
import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant import training_pipeline
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.ml.model.estimator import TargetValueMapping, ModelResolver
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataValidationArtifact


class ModelEvaluation:
    # simi to prediction pipeline

    def __init__(self, model_eval_config, data_validation_artifact, model_trainer_artifact):

        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self):

        try:
            # loading valid data
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path, low_memory=False)
            test_df = pd.read_csv(valid_test_file_path, low_memory=False)

            # concatenating the two dfs
            df = pd.concat([train_df, test_df])
            # defining the target col
            y_true = df[training_pipeline.TARGET_COLUMN]
            # Converting the target col to 1,0
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            # to get the features
            df.drop(training_pipeline.TARGET_COLUMN, axis=1, inplace=True)

            # loading the model
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path

            # firstly we need to check if the model is there or not, this is objected
            model_resolver = ModelResolver()

            # by default, we are accepting the model
            is_model_accepted = True

            # if my base model is not avail. then I don't have to do anything, just will create artifact
            if not model_resolver.is_model_exists():
                # if we do not have the model, we need to create the model artifact, do not have to do anything
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_file_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None)
                model_eval_report = model_evaluation_artifact.__dict__

                # save the report
                report_file_path = self.model_eval_config.report_file_path
                logging.info("Report file path: %s", report_file_path)
                # print("Report directory:", report_directory)
                write_yaml_file(report_file_path, model_eval_report)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            logging.info('Starting the next phase of evaluation')
            #but incase there is model avail then we need to find the latest model and then compare
            latest_model_path = model_resolver.get_best_model_path()
            #load latest model
            latest_model = load_object(latest_model_path)
            #loading the trained model
            train_model = load_object(train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            #getting classification score
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted = True

            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_file_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric
            )

            model_eval_report = model_evaluation_artifact.__dict__

            # save the report
            report_file_path = self.model_eval_config.report_file_path
            logging.info("Report file path:", report_file_path)
            #print("Report directory:", report_directory)
            write_yaml_file(report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)


