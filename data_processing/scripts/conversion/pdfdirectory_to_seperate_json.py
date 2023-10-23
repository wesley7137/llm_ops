import os
import json
from pdfminer.high_level import extract_text

def extract_pdf_info(pdf_path):
    # Extract text from the PDF file
    article_contents = extract_text(pdf_path)

    # Extract title as the first line of the document
    title = article_contents.split('\n')[0].strip()

    # Extract summary if it follows the "Summary" heading
    summary_start = article_contents.find("Summary")
    summary = "Summary not available"
    if summary_start != -1:
        summary_contents = article_contents[summary_start:].split('\n')[1:]
        summary = ' '.join([line.strip() for line in summary_contents if line.strip()])

    # You may need to manually provide or generate the subject
    subject = "Subject not available" # Modify as needed

    # Create JSON document
    json_document = {
        "title": title,
        "subject": subject,
        "summary": summary,
        "article_contents": article_contents
    }

    return json_document

# Directory containing PDF files
pdf_directory = "D:\\BCI\\pdfs_8_16_virtualbrainenv"

# Iterate through the files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        json_document = extract_pdf_info(pdf_path)

        # Save JSON document to a file
        json_filename = os.path.splitext(filename)[0] + ".json"
        with open(os.path.join(pdf_directory, json_filename), 'w') as json_file:
            json.dump(json_document, json_file)

        print(f"Processed {filename}")

print("Processing complete!")


# Directory containing PDF files




