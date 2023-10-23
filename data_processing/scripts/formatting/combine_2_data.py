import pandas as pd

# Load the datasets from JSON files
df1 = pd.read_json('C:\\Users\\wesla\\Downloads\\reasoning_examples.json')
df2 = pd.read_json('C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\Raw\\hypothesis_dataset_processed_for_bert.json')

# Rename the fields
df1 = df1.rename(columns={'problem': 'instruction', 'solution': 'output'})
df2 = df2.rename(columns={'source': 'instruction', 'target': 'output'})

# Combine the datasets
df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataset to a new JSON file
df.to_json('combined.json', orient='records', lines=True)
