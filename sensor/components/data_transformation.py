import os, sys
import numpy as np
import pandas as pd
from sensor.logger import logging
from sensor.ml.model import estimator
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sensor.exception import SensorException
from sensor.constant import training_pipeline
from sklearn.preprocessing import RobustScaler
from sensor.entity.config_entity import DataTransformationConfig
from sensor.utils.main_utils import save_numpy_array_data, save_object
from sensor.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact


class DataTransformation:

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
            :param data_validation_artifact: Output reference of data ingestion artifact stage
            :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise SensorException(e, sys)

    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls):
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ('SimpleImputer', simple_imputer),  # replace missing values with zero
                    ('RobustScaler', robust_scaler)  # keep every feature in same range and handle outlier
                ]
            )
            return preprocessor

        except Exception as e:
            raise SensorException(e, sys)

    def inititate_data_transformation(self):
        try:
            # reading the data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            # training dataframe
            input_feature_train_df = train_df.drop(columns=[training_pipeline.TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[training_pipeline.TARGET_COLUMN]
            # mapping the target features into 0,1
            target_feature_train_df = target_feature_train_df.replace(estimator.TargetValueMapping().to_dict())

            # testing dataframe
            input_feature_test_df = test_df.drop(columns=[training_pipeline.TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[training_pipeline.TARGET_COLUMN]
            # mapping the target features into 0,1
            target_feature_test_df = target_feature_test_df.replace(estimator.TargetValueMapping().to_dict())

            # scaling and imputing
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # upscaling
            smt = SMOTETomek(sampling_strategy='minority')

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

            # concatination of features and target, c_ = concatenation
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # save numpy array data (this is the funct)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # preparing data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
