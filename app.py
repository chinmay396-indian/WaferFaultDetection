from flask import Flask, jsonify, request, send_file, render_template
from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
import sys


app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to my application"


@app.route("/train")
def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()

        return "Training Completed"

    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods = ["POST","GET"])
def upload():
    try:
        if request.method == "POST":
            prediction_pipeline = PredictionPipeline()
            prediction_file = prediction_pipeline.run_pipeline()
            
            return send_file(prediction_file.final_predicted_data_path, as_attachment= True)
        
        else:
            return render_template("upload_file.html")

        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 5000, debug = True)