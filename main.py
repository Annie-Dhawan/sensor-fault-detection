import os, sys
import pandas as pd
from sensor.logger import logging
from uvicorn import run as app_run
from urllib.request import Request
from fastapi.responses import Response
from sensor.exception import SensorException
from sensor.constant import training_pipeline
from sensor.utils.main_utils import load_object
from starlette.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from sensor.constant.application import APP_HOST, APP_PORT
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        # else condition
        train_pipeline.run_pipeline()
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # get data from user csv file
        # convert csv file to dataframe
        df = pd.read_csv(file.file)
        model_resolver = ModelResolver(model_dir=training_pipeline.SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")

        # getting the best model path
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Convert DataFrame to JSON and return as response
        return df.to_json(orient='records')

    except Exception as e:
        raise Response(f"Error Occurred! {e}")



if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)

"""
if __name__ == '__main__':
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        logging.exception(e)"""
