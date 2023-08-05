import os

#defining common constant variables
TARGET_COLUMN = 'class'
PIPELINE_NAME = 'sensor'
ARTIFACT_DIR = 'artifact'
FILE_NAME = 'sensor.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
MODEL_FILE_NAME = 'model.pkl'
SAVED_MODEL_DIR = os.path.join("saved_models")
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLUMNS = "drop_columns"
DRIFT_THRESHOLD = 0.05

DATA_FILE_PATH = os.path.join('data', 'aps_failure_training_set1.csv')

'''
Data Ingestion related constants start with DATA_INGESTION_VAR_NAME
'''
#DATA_INGESTION_COLLECTION_NAME: str = 'sensor' ;this col is giving the name of the mongoDB collection name
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


'''
Data Validation related constants start with DATA_VALIDATION_VAR_NAME
'''

DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR: str = 'validated'
DATA_VALIDATION_INVALID_DIR: str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"


'''
Data Transformation related constants start with DATA_TRANSFORMATION_VAR_NAME
'''
DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"


"""
Model Trainer related constant start with MODEL_TRAINER_VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR: str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME: str = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05


"""
Model Evaluation related constants start with MODEL_EVALUATION_VAR_NAME
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.2
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"


"""
Model Pusher related constants start with MODEL_PUSHER_VAR_NAME
"""
MODEL_PUSHER_DIR_NAME: str = 'model_pusher'
MODEL_PUSHER_SAVED_MODEL_DIR: str = 'saved_model_dir'
