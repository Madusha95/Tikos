import pandas as pd
import webbrowser
import os

# Load MNIST train and test datasets
mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")

# Extract features only (exclude labels)
X_train = mnist_train.iloc[:, 1:].values
X_val = mnist_test.iloc[:, 1:].values

# Create DataFrames for Evidently with proper column names
column_names = [f"f{i}" for i in range(X_train.shape[1])]
train_df = pd.DataFrame(X_train, columns=column_names)
val_df = pd.DataFrame(X_val, columns=column_names)

# Ensure DataFrames contain numeric data
train_df = train_df.apply(pd.to_numeric, errors='coerce')
val_df = val_df.apply(pd.to_numeric, errors='coerce')

# ✅ Correct import for Evidently v0.7.0+
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Create the drift report
drift_report = Report(metrics=[DataDriftPreset()])

# Run the drift detection
drift_report.run(reference_data=train_df, current_data=val_df)

# Save the report as an HTML file
drift_report.save_html("drift_report.html")

print("Drift report saved as 'drift_report.html'")

# Open the report in a web browser (optional)
webbrowser.open('file://' + os.path.realpath("drift_report.html"))
