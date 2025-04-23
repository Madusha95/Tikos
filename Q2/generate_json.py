import json

# Create a list of zeros with length 728

pixels = [1] * 784

# Create the JSON object
json_object = {"pixels": pixels}

# Convert the Python object to a JSON string
json_string = json.dumps(json_object, indent=2)

# Print the resulting JSON string
print(json_string)
