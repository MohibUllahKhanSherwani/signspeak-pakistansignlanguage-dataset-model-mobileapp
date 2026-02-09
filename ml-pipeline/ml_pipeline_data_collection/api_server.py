import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import joblib
import os
from contextlib import asynccontextmanager
from typing import List


#Importing existing config
from actions_config import SEQUENCE_LENGTH, load_actions
MODEL_PATH = "action_model_baseline.h5"
ENCODER_PATH = "label_encoder_baseline.pkl"
#Import data model
from data_models import SequenceData

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    #load resources
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found at {ENCODER_PATH}")
    ml_models["model"] = load_model(MODEL_PATH)
    ml_models["encoder"] = joblib.load(ENCODER_PATH)
    print("Resources loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(lifespan = lifespan, title = "SignSpeak API")


@app.post("/predict")
async def predict(data: SequenceData):
    model = ml_models.get("model")
    encoder = ml_models.get("encoder")
    if not model or not encoder:
        raise HTTPException(status_code=500, detail="Model or encoder not loaded")
    # 1. convert intput to np arr
    sequence = np.array(data.landmarks)

    # 2. validate shape (excpected is (30, 126))
    expected_shape = (SEQUENCE_LENGTH, 126)
    if sequence.shape != expected_shape:
        raise HTTPException(status_code=400, detail=f"Invalid input shape. Expected {expected_shape}, got {sequence.shape}")

    #3. add batch dimensions (1, 30, 126)
    input_data = np.expand_dims(sequence, axis=0) 

    #4. predict 
    prediction = model.predict(input_data, verbose=0)[0]   
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    action_label = encoder.inverse_transform([predicted_index])[0]

    return {
        "action": action_label,
        "confidence": confidence,
        "all_probabilities": {
            label: float(prob)
            for label, prob in zip(encoder.classes_, prediction)
        }
    }
@app.get("/health")
def health_check():
    return {"status": "ok"}


    
