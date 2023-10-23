import pandas as pd
import json

# Load the data
with open('C:\\Users\\wesla\\Downloads\\wizard-vicuna-7B-uncensored\\Data\\processed\\deepmindrepositories.txt', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Add an ID column
df['id'] = range(1, len(df) + 1)

# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
