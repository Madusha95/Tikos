# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from model1 import DigitRecognizer  # Make sure this matches your model class file

app = FastAPI()

# Load the model only once at startup
model = DigitRecognizer()
model.load_state_dict(torch.load("digit_recognizer_model.pth", map_location=torch.device("cpu")))
model.eval()

# Input schema
class DigitInput(BaseModel):
    pixels: list  # Must be 784-length list of floats (flattened image)

@app.post("/predict")
def predict_digit(data: DigitInput):
    # Convert to tensor and normalize
    if len(data.pixels) != 784:
        return {"error": "Input must be a list of 784 pixel values."}

    input_array = np.array(data.pixels, dtype=np.float32) / 255.0
    input_tensor = torch.tensor(input_array).unsqueeze(0)  # shape: [1, 784]

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}
