import os
import json

def read_json_file(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)

# Directory containing JSON files
json_directory = "D:\\BCI\\pdfs_8_16_virtualbrainenv"

# List to store all JSON documents
all_documents = []

# Iterate through the files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith(".json"):
        json_path = os.path.join(json_directory, filename)
        json_document = read_json_file(json_path)
        all_documents.append(json_document)
        print(f"Processed {filename}")

# Save all documents to a single large JSON file
output_filename = "combined_documents.json"
with open(os.path.join(json_directory, output_filename), 'w') as json_file:
    json.dump(all_documents, json_file)

print("Concatenation complete!")
