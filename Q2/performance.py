import torch
import time
from model1 import DigitRecognizer
import pandas as pd
import numpy as np

# Load the trained model
model = DigitRecognizer()
model.load_state_dict(torch.load("digit_recognizer_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load the MNIST test dataset
mnist_test = pd.read_csv("mnist_test.csv")

# Extract features and labels
X_val = mnist_test.iloc[:, 1:].values  # All columns except the first one are features
Y_val = mnist_test.iloc[:, 0].values   # The first column is the label

# Sample data (X_val, Y_val must be defined)
sample_inputs = X_val[:100]  # Correctly take first 100 samples (not transposed)
sample_labels = Y_val[:100]

accurate = 0
latencies = []

for i in range(len(sample_inputs)):
    input_array = sample_inputs[i] / 255.0  # Normalize the input
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 784]

    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    end_time = time.time()

    latency = end_time - start_time
    latencies.append(latency)

    if prediction == sample_labels[i]:
        accurate += 1

accuracy = accurate / len(sample_labels)
avg_latency = sum(latencies) / len(latencies)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Average Latency: {avg_latency*1000:.2f} ms")
print(f"Total Inference Time: {sum(latencies)*1000:.2f} ms")