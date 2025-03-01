from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model in advance from the file:
model_path = "model.joblib"
model = joblib.load(model_path)  # Load the trained model

# Input data schema (4 features)
class InputData(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.post("/predict")
def predict(data: InputData):
    """
    We receive 4 parameters, make a prediction,
    and return the result.
    """
    # Form a "table" with a single observation:
    new_data = [[
        data.variance,
        data.skewness,
        data.curtosis,
        data.entropy
    ]]

    # Call predict
    prediction = model.predict(new_data)
    # Return the class as a number (or you can return a string)
    return {"prediction": int(prediction[0])}