import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from src.config import Config
from src.logger import setup_logger
import logging
import numpy as np
import pandas as pd

# Initialize config and logger
config = Config()
logging_config = config.get_logging_config()
logger = setup_logger(
    name="banknote_api",
    log_file=logging_config["log_file"],
    level=getattr(logging, logging_config["level"])
)

# Get model path from config
model_path = config.get_train_config()['model_path']

app = FastAPI(
    title="Banknote Authentication API",
    description="API for predicting banknote authenticity",
    version="1.0.0"
)

# Load the model in advance from the file
logger.info(f"Loading model from {model_path}")
try:
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Input data schema (4 features)
class InputData(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.post("/predict")
def predict(data: InputData):
    """
    Make a prediction for banknote authenticity.
    
    Args:
        data: Input features (variance, skewness, curtosis, entropy)
        
    Returns:
        Dictionary containing prediction (0 for authentic, 1 for counterfeit)
    """
    logger.info(f"Received prediction request with features: {data.dict()}")
    
    try:

        new_data = pd.DataFrame([[
            data.variance,
            data.skewness,
            data.curtosis,
            data.entropy
        ]])

        # Call predict
        prediction = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)
        
        result = {
            "prediction": int(prediction[0]),
            "probability": float(np.max(prediction_proba[0]))
        }
        
        logger.info(f"Prediction made: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    logger.info("Health check requested")
    return {"status": "healthy"}

# The entry point for running on port 8080
if __name__ == "__main__":
    logger.info("Starting API server")
    uvicorn.run("server:app", host="0.0.0.0", port=8080)