import json

def convert_description_to_question(description):
    # Simple heuristic to turn a task description into a question
    question = f"What is involved in {description.lower()}?"
    return question

input_file_path = "D:\\BCI\\Data\\master_model_training_PROCESSED_fixed.json"
output_file_path = "D:\\BCI\\Data\\master_model_training_PROCESSED_fixed_questions.json"  # New file path

# Read the JSON objects
print("Reading JSON objects from file...")
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Process the objects
print("Processing objects...")
updated_data = []
for i, obj in enumerate(data):
    if isinstance(obj, dict) and "task_description" in obj:
        task_description = obj["task_description"]
        question = convert_description_to_question(task_description)

        # Log original task description and converted question for debugging
        print(f"Original Task Description: {task_description}")
        print(f"Converted Question: {question}")

        obj["task_description"] = question
        updated_data.append(obj)

# Write the updated data back to the new JSON file
print("Writing updated data to the new JSON file...")
with open(output_file_path, 'w') as file:
    json.dump(updated_data, file, separators=(',', ':'))

print("Task descriptions have been successfully converted to questions and saved to a new file.")
