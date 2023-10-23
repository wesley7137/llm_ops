import os
import PyPDF2

# Directory containing the PDFs
pdf_dir = input("Enter the directory containing the PDFs: ")

# Output text file
output_file = input("Enter the name of the output text file (e.g., output.txt): ")

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate over the files in the directory
    for filename in os.listdir(pdf_dir):
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            # Construct the full file path
            pdf_path = os.path.join(pdf_dir, filename)
            # Write a line to the output file indicating the start of a new file's content
            outfile.write(f"\n--- Start of Document {filename} ---\n")
            # Open the PDF file in read-binary mode
            with open(pdf_path, 'rb') as pdffile:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdffile)
                # Iterate over the pages in the PDF (starting from page 0)
                for page_num in range(len(pdf_reader.pages)):
                    # Extract the text from the page
                    text = pdf_reader.pages[page_num].extract_text()
                    # Write the text to the output file, adding a newline character at the end
                    outfile.write(text + '\n')
            # Write a line to the output file indicating the end of the current file's content
            outfile.write(f"\n--- End of {filename} ---\n")

print(f"Text extracted from PDFs and written to {output_file}")
