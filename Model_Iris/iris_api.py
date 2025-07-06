from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("iris_best_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
label_encoder = joblib.load("iris_label_encoder.pkl")

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict")
def predict_species(data: IrisInput):
    features = np.array([
        [data.SepalLengthCm, data.SepalWidthCm, data.PetalLengthCm, data.PetalWidthCm]
    ])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    label = label_encoder.inverse_transform(prediction)[0]
    return {"prediction": label}
