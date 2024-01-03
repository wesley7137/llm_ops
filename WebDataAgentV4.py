
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import random
import requests
from bs4 import BeautifulSoup
import yaml
import os
import datetime
from duckduckgo_search import DDGS
import torch
import sseclient  # Ensure sseclient-py is installed
import json
import re
from collections import Counter
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
from urllib.parse import urlparse
import tldextract
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging

objective = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_completion(prompt, max_tokens=1024, temperature=1, top_p=0.9, seed=10):
    # Ensure this URL is correct and matches the server's completions endpoint
    url = "http://127.0.0.1:5000/v1/completions"

    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": True
    }

    # Make the post request and handle the streamed response
    stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    completion = ''
    for event in client.events():
        payload = json.loads(event.data)
        completion += payload['choices'][0]['text']

    return completion
            


class SerpApiSearch:
    def __init__(self, api_key):
        self.api_key = ""  # Replace with your SerpApi key
        self.base_url = "https://serpapi.com/search"

    def search(self, query, location=None, google_domain="google.com", num_results=10):
        params = {
            "engine": "google",
            "q": query,
            "google_domain": google_domain,
            "num": num_results,
            "api_key": self.api_key
        }
        if location:
            params["location"] = location

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error in SerpApi search: {e}")
            return None

# Example usage


USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0"
    # List of user agents
]


def scrape_website(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()  # or HTMLSession() for JavaScript content
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    try:
        response = http.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Implement logic to handle dynamic JavaScript content if needed
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        save_data(data=paragraphs, objective=objective, prefix="file")

        return ' '.join(paragraphs)
    except requests.exceptions.HTTPError as e:
        # Handling specific HTTP errors
        return handle_http_error(e, url)
    except requests.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None


def handle_http_error(e, url):
    if e.response.status_code == 403:
        logging.error(f"Access Denied when scraping {url}. It might be blocking scrapers.")
        # Implement logic for CAPTCHA or proxy rotation if needed
    else:
        logging.error(f"HTTPError when scraping {url}: {e}")
    return None


def save_data(data, objective, prefix="file"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    objective_clean = objective[:15].replace(" ", "_")
    file_name = f"{prefix}_{objective_clean}_{timestamp}.txt"
    dir_path = "data"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w') as file:
        if isinstance(data, list):
            data = '\n'.join(data)  # Convert list to string
        file.write(data)
        logging.info(f"Saved data to {file_path}")
        
        
# Function to chunk content into 1000-token pieces
def chunk_content(content, chunk_size=2000):
    tokens = word_tokenize(content)
    for i in range(0, len(tokens), chunk_size):
        yield ' '.join(tokens[i:i + chunk_size])

# Modified generate_summary function
def generate_summary(content, objective=objective, max_tokens=1024, temperature=1, top_p=0.9, seed=10):
    url = "http://127.0.0.1:5000/v1/completions"  # Use the global keyword to indicate you want to use the global variable
    summaries = []
    for chunk in chunk_content(content):
        system_message = """You are a summarizer tasked with creating summaries based on a specified methodology. Your key activities include thoroughly understanding the specified methodology, identifying the main points and key details in the given content, and condensing the information into a concise summary that accurately reflects the original content. It is important to avoid any risks such as misinterpreting the methodology, omitting crucial information, or distorting the original meaning. Use clear and specific language to convey your request, ensuring that the summary is coherent, well-organized, and effectively communicates the main ideas of the original content. Use bullet points or lists where necessary. Use the following method to summarize: """

        prompt = system_message + chunk
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True
        }
        stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
        client = sseclient.SSEClient(stream_response)

        summary_chunk = ''
        for event in client.events():
            payload = json.loads(event.data)
            summary_chunk += payload['choices'][0]['text']
        
        summaries.append(summary_chunk)
        # Optionally save each chunk's summary before proceeding
        
    save_data(data=summaries, objective=objective, prefix="summary")

    return ' '.join(summaries)



def get_objective():
    global objective  # Use the global keyword to indicate you want to use the global variable

    return input("Enter the Objective: ")

def extract_keywords(objective, max_keywords=6):

    # Remove punctuation and lower the case
    clean_text = re.sub(r'[^\w\s]', '', objective).lower()
    # Tokenize the text into words
    words = word_tokenize(clean_text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    # Stem the words
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    # Extract Nouns and Adjectives
    tagged_words = pos_tag(stemmed_words)
    keywords = [word for word, tag in tagged_words if tag in ('NN', 'NNS', 'JJ')]
    # Count and sort the keywords
    word_freq = Counter(keywords)
    sorted_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:max_keywords]

    return sorted_keywords

                                                                                                                                                                        
def get_content_bs4(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except requests.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None

def is_summary_relevant(summary, objective):
    # Implement a simple check for relevance (can be refined later)
    relevant_terms = set(objective.lower().split())
    summary_terms = set(summary.lower().split())
    return bool(relevant_terms.intersection(summary_terms))

# Maximum number of successful scrapes required
MIN_SUCCESSFUL_SCRAPES = 5

def save_url_contents(contents, filename="url_content.txt"):
    with open(filename, "a") as file:
        file.write(contents + "\n\n")

# Function to read content from a file
def read_file(filename):
    with open(filename, "r") as file:
        return file.read()


def summarize_from_file(filename="url_content.txt"):
    content = read_file(filename)
    summarized_content = generate_summary(content, objective)

    with open("generated_summaries.txt", "w") as file:
        file.write(summarized_content)

def scrape_and_save_contents(search_results, filename="url_content.txt"):
    visited_domains = []
    successful_scrapes = 0

    for result in search_results['organic_results']:
        if successful_scrapes >= MIN_SUCCESSFUL_SCRAPES:
            break

        url = result.get('link', '')
        parsed_url = urlparse(url)
        domain_info = tldextract.extract(parsed_url.netloc)
        main_domain = '.'.join([domain_info.domain, domain_info.suffix])

        if not main_domain or main_domain in visited_domains:
            logging.info(f"Skipping {url} as it is a duplicate domain or missing URL")
            continue

        visited_domains.append(main_domain)
        content = scrape_website(url)
        
        if content:
            save_url_contents(content, filename)
            successful_scrapes += 1
            logging.info(f"Saved content for {url}")

# Function to save URL contents to a file
def save_url_contents(contents, filename):
    with open(filename, "a") as file:
        file.write(contents + "\n\n")

# Main function modified to call the new functions
def main():
    global objective
    serp_api_search = SerpApiSearch(api_key="SERPAPI_KEY")

    objective = get_objective()
    sorted_keywords = extract_keywords(objective, max_keywords=10)
    search_results = serp_api_search.search(" ".join(sorted_keywords), num_results=10)

    if not search_results or 'organic_results' not in search_results:
        logging.error("No valid search results found")
        return

    # Step 1: Scrape and save URL contents
    scrape_and_save_contents(search_results)

    # Step 2: Summarize saved content and save summaries
    summarize_from_file()

if __name__ == "__main__":
    main()
