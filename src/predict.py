# src/predict.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Заранее загружаем модель из файла:
model_path = "model.joblib"
model = joblib.load(model_path)  # подгрузит обученную модель

# Входная схема данных (4 признака)
class InputData(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.post("/predict")
def predict(data: InputData):
    """
    Получаем на вход 4 параметра, делаем предсказание
    и возвращаем результат.
    """
    # Формируем "табличку" с одним наблюдением:
    new_data = [[
        data.variance, 
        data.skewness, 
        data.curtosis, 
        data.entropy
    ]]

    # Вызываем predict
    prediction = model.predict(new_data)
    # Возвращаем класс как число (или можете отдать строку)
    return {"prediction": int(prediction[0])}