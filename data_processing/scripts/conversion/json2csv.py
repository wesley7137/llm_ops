

# Open the json file
import json
import pandas as pd

# Path to the JSON file
json_file_path = "C:\\Users\\wesla\\oobabooga_windows\\text-generation-webui\\training\\datasets\\neuro-alpaca-data-ooga.json"

# Read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
csv_file_path = "C:\\Users\\wesla\\neuro-alpaca-data-ooga.csv"
df.to_csv(csv_file_path, index=False)
