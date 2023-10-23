
import pandas as pd
import json

# Read the JSON file
with open('C:\\Users\\wesla\\addiction_counseling_synthetic.json', 'r') as file:
    data = json.load(file)

# Format the data into a single text column
formatted_data = [
    {"text": f"### Input: {item['input']} ### Response: {item['output']}"} 
    for item in data
]

# Convert to a DataFrame
df = pd.DataFrame(formatted_data)

# Save as CSV
df.to_csv('formatted_data.csv', index=False)