import json

# Path to the input JSON file
input_json_file_path = "C:\\Users\\wesla\\Downloads\\medical_qa_combined.json"

# Path to the output JSON file
output_json_file_path = 'medicalQAcombined_processed.json'

# List to store the input and output data
input_output_data = []

# Read the input JSON file
with open(input_json_file_path, 'r', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)
    for entry in data:
        input_text = entry['input']
        output_text = entry['output']
        input_output_data.append({'input': input_text, 'output': output_text})

# Write to the output JSON file
with open(output_json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(input_output_data, jsonfile, ensure_ascii=False, indent=4)

print(f"Input and output extracted and saved as {output_json_file_path}")
