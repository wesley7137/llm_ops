import re

def process_text(text):
    # Split the text into sections based on the headings
    sections = re.split(r'\b(Abstract|Contents|ACKNOWLEDGMENTS|REFERENCES)\b', text, flags=re.IGNORECASE)
    
    if len(sections) < 8:
        return None  # The document structure is not as expected

    # The main text is the third section of the split
    main_text = sections[4]

    return main_text.strip()

# Load your text data (replace this with the actual path to your file)
with open('C:\\Users\\wesla\\OneDrive\\Desktop\\Wizard-Vicuna-7B-Uncensored\\wizard-vicuna-7B-uncensored22\\Data\\quantum_research\\quantum_data\\quantum_research.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Process the text data
processed_data = process_text(text_data)

# Save the processed data (replace this with the path where you want to save the processed data)
with open('C:\\Users\\wesla\\quantum_research.txt', 'w') as file:
    file.write(processed_data)
