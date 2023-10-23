import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('conceptofmind/cot_submix_original')

# Convert the Dataset object to a list of dictionaries
dataset = dataset['train'].to_dict()

# Check if the dataset is loaded correctly
print(f'Dataset loaded: {len(dataset)} examples')

# Function to format the data
def format_data(data):
    formatted_data = []
    for example in data:
        # Replace the placeholders with actual instruction and output
        formatted_example = {
            "instruction,output": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: %s\n\nASSISTANT: %s" % (example['inputs'], example['targets'])
        }
        formatted_data.append(formatted_example)
    return formatted_data

# Format the dataset
formatted_dataset = format_data(dataset)

# Check if the data is formatted correctly
print(f'Data formatted: {len(formatted_dataset)} examples')

# Save the formatted dataset to a JSON file
with open('formatted_dataset.json', 'w') as f:
    json.dump(formatted_dataset, f)
