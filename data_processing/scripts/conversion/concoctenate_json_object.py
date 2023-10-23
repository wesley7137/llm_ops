import json

input_file_path = "D:\\BCI\\Data\\master_model_training_PROCESSED_fixed_questions.json"
output_file_path = "D:\\BCI\\Data\\master_model_training_PROCESSED_fixed_instructions_combined.json"  # New file path

# Read the JSON objects
print("Reading JSON objects from file...")
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Process the objects
print("Processing objects...")
updated_data = []
for i, obj in enumerate(data):
    if isinstance(obj, dict) and "instructions" in obj:
        # Combine the instructions into a single string
        instructions_string = ". ".join(obj["instructions"])
        obj["instructions"] = instructions_string
        updated_data.append(obj)
    print(f"Processed {i + 1} out of {len(data)} objects.")

# Write the updated data back to the new JSON file
print("Writing updated data to the new JSON file...")
with open(output_file_path, 'w') as file:
    json.dump(updated_data, file, separators=(',', ':'))

print("Instructions have been successfully combined into single strings and saved to a new file.")
