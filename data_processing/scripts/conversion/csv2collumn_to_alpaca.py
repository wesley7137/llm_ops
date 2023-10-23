
import pandas as pd

# Load your dataset
df = pd.read_json('C:\\Users\\wesla\\medicalQAcombined_processed.json')

# Combine the 'input' and 'output' columns
df['text'] = ' ### Human: ' + df['input'].astype(str) + ' ### Assistant: ' + df['output'].astype(str)

# Save the new DataFrame to a CSV file
df.to_csv('medicalQAcombined_processed.csv', index=False)

