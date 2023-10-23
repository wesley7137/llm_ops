
import json
import pandas as pd

# Read the file and load it into prev_examples
with open('C:\\Users\\wesla\\modified_qa_data_BCI_OMGscript.txt', 'r') as f:
    prev_examples = json.load(f)  # Assumes the file contains a JSON-formatted list

# Initialize lists to store prompts and responses
prompts = []
responses = []

# Loop through the prev_examples list and print out the index where the exception occurs
for i, example in enumerate(prev_examples):
    try:
        split_example = example.split('-----------')
        prompts.append(split_example[1].strip())
        responses.append(split_example[3].strip())
    except Exception as e:
        print(f"Error occurred at index {i}: {e}")
        print(f"Problematic string: {example[:100]}...")  # prints the first 100 characters of the problematic string

# Create a DataFrame
try:
    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f'There are {len(df)} successfully-generated examples. Here are the first few:')
    print(df.head())
except ValueError as ve:
    print(f"ValueError: {ve}")