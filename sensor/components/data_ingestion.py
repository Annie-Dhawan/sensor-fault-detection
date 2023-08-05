import sys, os
import numpy as np
import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant import training_pipeline
from sensor.utils.main_utils import read_yaml_file
from sklearn.model_selection import train_test_split
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        # this will give us the file path
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)

        except Exception as e:
            raise SensorException(e, sys)

    def export_data_into_feature_store(self):
        """
        Exporting data into feature store
        :return: DataFrame
        """
        try:
            logging.info('Exporting data into feature store')
            df = pd.read_csv(training_pipeline.DATA_FILE_PATH)
            df.replace({'na': np.nan}, inplace=True)

            # firstly we are creating a feature store dir and then we gotta upload the sensor.csv in it
            feature_file_path = self.data_ingestion_config.feature_store_file_path  # got file path

            # creating the folder, dirname is to get the file path, this is only to get the dir name
            dir_path = os.path.dirname(feature_file_path)

            # creating a feature store file path if it has already not being created
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_file_path, index=False, header=True)
            logging.info("Exporting data into feature store is completed")
            return df

        except Exception as e:
            raise SensorException(e, sys)

    def split_data_as_train_test(self, dataframe):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            #getting the file path
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            #creating a folder ingested
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise SensorException(e, sys)

    def inititate_data_ingestion(self):
        """
        :return: DataIngestionArtifact
        """
        try:
            df = self.export_data_into_feature_store()
            df = df.drop(self._schema_config["drop_columns"], axis=1)

            self.split_data_as_train_test(dataframe=df)
            #passing the train and test
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)






