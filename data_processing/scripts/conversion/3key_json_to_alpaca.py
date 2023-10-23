import pandas as pd
import json

# Load your dataset from JSON file
df = pd.read_json('D:\\BCI\\Data\\qa_dataset.json', encoding='latin-1')

# Concatenate the values with the specified format
instruction_input_output = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n### Instruction:\n"
    + df['context'].str.cat(sep='\n')
    + "\n\n### Input:\n"
    + df['question'].str.cat(sep='\n')
    + "\n\n### Response:\n"
    + df['answer'].str.cat(sep='\n')
)

# Create the required JSON structure
result = {
    "instruction,input,output": instruction_input_output
}

# Save the modified data to a new JSON file
with open('modified_data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
