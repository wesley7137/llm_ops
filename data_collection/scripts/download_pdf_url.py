import os
import glob
from pdfminer.high_level import extract_text

def pdf_to_text(directory):
    # Get all PDF files in the directory
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
    
    for pdf_file in pdf_files:
        # Generate the corresponding text file path
        text_file = os.path.splitext(pdf_file)[0] + '.txt'
        
        # Convert PDF to text
        text = extract_text(pdf_file)
        
        # Save the text to a file
        with open(text_file, 'w', encoding='utf-8') as file:
            file.write(text)
        
        print(f'Successfully converted {pdf_file} to {text_file}')

# Example usage
directory_path = 'C:\\Users\\wesla\\OneDrive\\Desktop\\PDF_DL'  # Replace with your directory path
pdf_to_text(directory_path)
