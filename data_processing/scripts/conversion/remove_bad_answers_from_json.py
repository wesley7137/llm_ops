
import json

# Path to your JSON file
file_path = "C:\\Users\\wesla\\qa_dataset.json"

# List of phrases that indicate an unwanted answer
unwanted_phrases = [
    "The text does not provide ",
    "There is no mention of  ",
    "Unfortunately, ",
    "there is no direct information",
    "it cannot be determined",
    "The text does not ",
    "I'm sorry",
    "Apologies for the confusion",
    "it is impossible to determine",
    "Sorry",
    "The question is unclear",
    "question is not provided",
    "it is not possible to determine",
    "it does not provide",
    "question is irrelevant",
    "does not provide",
    "does not discuss",
    "I apologize"
]

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to determine if an answer should be filtered
def should_filter(answer):
    return any(phrase in answer for phrase in unwanted_phrases)

# Filter the data
filtered_data = [entry for entry in data if not should_filter(entry['answer'])]

# Save the filtered data back to a JSON file
with open(file_path, 'w') as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered {len(data) - len(filtered_data)} entries with undesired answers.")
