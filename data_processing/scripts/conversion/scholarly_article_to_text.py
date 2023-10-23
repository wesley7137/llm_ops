
#!pip install requests pdfplumber feedparser

import os
import requests
import pdfplumber
import feedparser


ARXIV_API = "http://export.arxiv.org/api/query"
PDF_DIR = "pdfs"
OUTPUT_FILE = "output.txt"
MAX_RESULTS = 20

search_terms = ["artificial intelligence", "quantum physics", "data structures and algorithms "]

def get_arxiv_articles(search_term, max_results=MAX_RESULTS):
    response = requests.get(ARXIV_API, params={
        "search_query": search_term,
        "start": 0,
        "max_results": max_results
    })

    return feedparser.parse(response.content)

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(filename):
    with pdfplumber.open(filename) as pdf:
        return '\n'.join(page.extract_text() for page in pdf.pages)

if __name__ == "__main__":
    # Create a directory to store the PDFs
    os.makedirs(PDF_DIR, exist_ok=True)

    # Open the text file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for term in search_terms:
            print(f"Processing search term: {term}")
            articles = get_arxiv_articles(term)
            for i, article in enumerate(articles.entries):
                # Check if PDF link is available
                for link in article.links:
                    if link.type == 'application/pdf':
                        url = link.href
                        break
                else:
                    url = article.link

                # Download the PDF
                filename = f'{PDF_DIR}/document_{term.replace(" ", "_")}_{i}.pdf'
                print(f"Downloading PDF from {url} to {filename}")
                try:
                    download_pdf(url, filename)
                except Exception as e:
                    print(f"Error downloading PDF: {e}")
                    continue

                # Extract the text
                print(f"Extracting text from {filename}")
                try:
                    text = extract_text_from_pdf(filename)
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    continue

                # Write the text to the file with separators
                f.write(f'--- Start of Document {term.replace(" ", "_")}_{i} ---\n')
                f.write(text)
                f.write(f'\n--- End of Document {term.replace(" ", "_")}_{i} ---\n')
