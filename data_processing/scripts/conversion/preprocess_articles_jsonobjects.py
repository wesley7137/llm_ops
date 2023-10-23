import json
import re

def process_text(text):
    # Split the text into sections based on the headings
    sections = re.split(r'\b(Abstract|Contents|ACKNOWLEDGMENTS|REFERENCES)\b', text, flags=re.IGNORECASE)
    
    if len(sections) < 8:
        return None  # The document structure is not as expected

    # Extract the different parts of the document
    title = sections[0].strip()
    abstract = sections[2].strip()
    contents = sections[4].strip()

    # Create a dictionary to store the data
    data = {
        'title': title,
        'abstract': abstract,
        'contents': contents
    }

    return data

# Load your text data (replace this with the actual path to your file)
with open('C:\\Users\\wesla\\OneDrive\\Desktop\\Wizard-Vicuna-7B-Uncensored\\wizard-vicuna-7B-uncensored22\\Data\\quantum_research\\quantum_data\\quantum_research.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Process the text data
processed_data = process_text(text_data)

# Save the processed data as a JSON file (replace this with the path where you want to save the processed data)
with open('c:\\Users\\wesla\\OneDrive\\Desktop\\Wizard-Vicuna-7B-Uncensored\\wizard-vicuna-7B-uncensored22\\Data\\processed\\quantum_research.json', 'w') as file:
    json.dump(processed_data, file)

