import requests
import os
import pdfplumber
import time

# Function to download a PDF from a URL
def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Function to extract text from a PDF file
def extract_text_from_pdf(filename):
    with pdfplumber.open(filename) as pdf:
        return '\n'.join(page.extract_text() for page in pdf.pages)

# Create a directory for the PDF files
os.makedirs('pdfs', exist_ok=True)

# List of PDF URLs
pdf_urls = ['https://arxiv.org/pdf/2308.04030.pdf']


# Open the output file
with open('output.txt', 'w', encoding='utf-8') as f_output:
    # Loop over the URLs
    for i, url in enumerate(pdf_urls):
        # Define a filename for each PDF
        filename = f"pdfs/document_{i}.pdf"

        # Download the PDF
        print(f"Downloading {url} to {filename}...")
        download_pdf(url, filename)

        # Extract the text from the PDF
        print(f"Extracting text from {filename}...")
        text = extract_text_from_pdf(filename)

        # Write the extracted text into the output file
        f_output.write(text)
        f_output.write("\n---\n")

        # Sleep for 10 seconds to prevent rate-limiting
        time.sleep(10)

print("All files have been processed.")
