import os, sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant import training_pipeline


class TargetValueMapping:
    """
    We are doing label encoding here, this is a self made but we can also use endcoder class too
    """

    def __init__(self):
        self.pos: int = 1
        self.neg: int = 0

    def to_dict(self):
        """
        Mapping the target values in dict
        :return: dict like {'neg':0,'pos':1}
        """
        return self.__dict__

    def reverse_mapping(self):
        """
            It will tell us whihc is what
            :return: dict like {0:'neg',1:'pos'}
        """
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class SensorModel:

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat

        except Exception as e:
            raise SensorException(e, sys)


class ModelResolver:

    def __init__(self, model_dir=training_pipeline.SAVED_MODEL_DIR):

        try:
            self.model_dir = model_dir

        except Exception as e:
            raise SensorException(e, sys)

    def get_best_model_path(self):

        """
        Here all the best models will be saved
         => Flow
         1. saved model dir
          1.1. Timestamp
             1.1.1. model.pkl
        """

        try:
            # converting timestamps to int and also applying max in the next step so that we can get the max value as the current best model
            timestamps = map(int, os.listdir(self.model_dir))

            latest_timestamp = max(timestamps)

            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", training_pipeline.MODEL_FILE_NAME)

            return latest_model_path

        except Exception as e:
            raise SensorException(e, sys)

    def is_model_exists(self):
        """
            Checking if the folder is there or not
            :return: bool
        """
        try:
            # 1: checking if the folder is there or not
            if not os.path.exists(self.model_dir):
                return False

            # 2:checking if the folder is there but not the timestamp folder then return false
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False

            # 3: if the folder,timestamp folder is availabe then we are calling the function to get latest model file
            latest_model_path = self.get_best_model_path()

            # 4: if the file is not avail then return false
            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e
