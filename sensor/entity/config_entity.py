# it is all about the input dirs or files
import os
from datetime import datetime
from sensor.constant import training_pipeline


class TrainingPipelineConfig:
    """
    1. Artifact_dir
     1.1. Timestamp folder
    We are creating artifact folder here. This timestamp will seggregate many folders made
    since we will run pipline many times.
    The artifact folder will have all the folders like data validation etc.
    """

    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,
                                              timestamp)  # we created artifact dir and then after that we are creating timestamp
        # /path/to/artifacts/2023-07-06_1345 like this
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    1. Artifact_dir
     1.1. Timestamp folder
        1.1.1. Data Ingestion folder
            1.1.1.1 Feature Store folder
                  1.1.1.1.1. sensor.csv
            1.1.1.2. ingested folder
                  1.1.1.2.1. train.csv
                  1.1.1.2.2. test.csv
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # We are making data ingestion folder
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                    training_pipeline.DATA_INGESTION_DIR_NAME)

        # we are going to now store the sensor.csv in Feature Store Folder
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir,
                                                         training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                                                         training_pipeline.FILE_NAME)

        # we are now going to make ingested folder and then make train,test csv
        self.training_file_path: str = os.path.join(self.data_ingestion_dir,
                                                    training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                    training_pipeline.TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir,
                                                   training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                   training_pipeline.TEST_FILE_NAME)

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


class DataValidationConfig:
    """
    1. Artifact_dir
     1.1. Timestamp folder
       1.1.1. Data Validation folder
           1.1.1.1. Valid Data dir(training)
                1.1.1.1.1. Train.csv
                1.1.1.1.2. Test.csv
           1.1.1.2. Valid Data dir(training)
                1.1.1.2.1. None
                1.1.1.2.2. None

    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                     training_pipeline.DATA_VALIDATION_DIR_NAME)

        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)

        self.invalid_data_dir: str = os.path.join(self.data_validation_dir,
                                                  training_pipeline.DATA_VALIDATION_INVALID_DIR)

        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)

        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)

        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)

        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)

        self.drift_report_file_path: str = os.path.join(self.data_validation_dir,
                                                        training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                        training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)


class DataTransformationConfig:
    """
    1. Artifact_dir
     1.1 Timestamp folder
        1.1.1. Data Transformation Folder
             1.1.1.1. Transformed_dir
                    1.1.1.1.1 train.npy
                    1.1.1.1.2 test.npy
             1.1.1.2. Transformed Object
                    1.1.1.2.1. preprocessing.pkl
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                         training_pipeline.DATA_TRANSFORMATION_DIR_NAME)

        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,
                                                             training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                             training_pipeline.TRAIN_FILE_NAME.replace('csv', 'npy'))

        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,
                                                            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                            training_pipeline.TEST_FILE_NAME.replace('csv', 'npy'))

        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir,
                                                              training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                              training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)


class ModelTrainerConfig:
    """
    1. Artifact_dir
     1.1. Timestamp Folder
        1.1.1. Model Trainer Folder
             1.1.1.1 Trained Model Dir
                    1.1.1.1.1. model.pkl
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                   training_pipeline.MODEL_TRAINER_DIR_NAME)

        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir,
                                                         training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
                                                         training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME)

        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD


class ModelEvaluationConfig:
    """
    1. Artifact_dir
     1.1. Timestamp folder
        1.1.1. Model Evaluation folder
             1.1.1.1. report.yaml
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                      training_pipeline.MODEL_EVALUATION_DIR_NAME)

        self.report_file_path: str = os.path.join(self.model_evaluation_dir,
                                                  training_pipeline.MODEL_EVALUATION_REPORT_NAME)

        self.change_threshold: float = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    """
    1. Artifact_dir
     1.1. Timestamp folder
        1.1.1 Model Pusher


    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_pusher_dir_file_path = os.path.join(training_pipeline_config.artifact_dir,
                                                       training_pipeline.MODEL_PUSHER_DIR_NAME)

        self.model_file_path = os.path.join(self.model_pusher_dir_file_path,
                                            training_pipeline.MODEL_FILE_NAME)

        timestamp = round(datetime.now().timestamp())

        self.saved_model_path = os.path.join(
            training_pipeline.SAVED_MODEL_DIR,
            f"{timestamp}",
            training_pipeline.MODEL_FILE_NAME)
