import csv
import json

# Path to the CSV file
csv_file_path = 'path_to_your_file.csv'

# Path to the output JSON file
json_file_path = 'output_file.json'

# List to store the questions and answers
qa_data = []

# Read the CSV file
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if there is one
    for row in reader:
        input_text, output_text = row
        qa_data.append({'question': input_text, 'answer': output_text})

# Write to the JSON file
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(qa_data, jsonfile, ensure_ascii=False, indent=4)

print(f"CSV file converted to JSON and saved as {json_file_path}")
