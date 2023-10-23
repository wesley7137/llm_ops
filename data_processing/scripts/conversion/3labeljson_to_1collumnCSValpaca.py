import pandas as pd
import json

# Read the JSON file
with open("C:\\Users\\wesla\\qa_dataset.json") as f:
    data = json.load(f)

# Prepare the CSV content
csv_content = []
for entry in data:
    instruction = entry['context'] + ' ' + entry['question']
    response = entry['answer']
    line = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    csv_content.append(line)

# Convert to DataFrame
df = pd.DataFrame(csv_content, columns=['question_answer'])

# Save to CSV file
df.to_csv('neor-llama-finetune-autotrain.csv', index=False, encoding='utf-8')
