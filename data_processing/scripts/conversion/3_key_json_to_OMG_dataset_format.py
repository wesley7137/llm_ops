import pandas as pd
import json

# Load your dataset from JSON file
df = pd.read_json('D:\\BCI\\Data\\qa_dataset.json', encoding='utf-8')

# Generate the formatted string by iterating over the DataFrame rows
formatted_data = []
for index, row in df.iterrows():
    formatted_entry = (
        "prompt\n-----------\n" + row['question'] + "\n-----------\n\n" +
        "response\n-----------\n" + row['answer']
    )
    formatted_data.append(formatted_entry)

# Concatenate all formatted entries into a single string, separated by '", "'
final_output = '", "'.join(formatted_data)

# Wrap the string in double quotes
final_output = '"' + final_output + '"'

# Save the modified data to a new JSON file
with open('modified_qa_data.json', 'w', encoding='utf-8') as f:
    json.dump({"instruction,input,output": final_output}, f, ensure_ascii=False, indent=4)
