import time
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from tqdm import tqdm
from ratelimiter import RateLimiter

rate_limiter = RateLimiter(max_calls=10, period=1)  # 10 requests per second

def scrape_documentation():
    doc_url = input('Please enter the URL of the website you want to scrape: ')

    base_url = '/'.join(doc_url.split('/')[:3])
    all_pages = set()

    stack = [doc_url]
    while stack:
        url = stack.pop()
        if url not in all_pages:
            all_pages.add(url)
            try:
                with rate_limiter:
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, 'lxml')
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and not href.startswith('#') and href.startswith('http'):
                        full_url = urljoin(base_url, href)
                        stack.append(full_url)
            except Exception as e:
                print(f'Error while accessing {url}: {e}')

    all_pdf_links = []
    for url in tqdm(all_pages, desc='Scraping pages', unit='page'):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith('https://arxiv.org/pdf/'):
                    print(f'Found PDF link: {href}')
                    save_pdf_link(href)
                    rate_limit_counter += 1
                    if rate_limit_counter % 50 == 0:
                        time.sleep(5)  # Rate limiter: pause for 2 seconds
        except Exception as e:
            print(f'Error while accessing {url}: {e}')

    if all_pdf_links:
        save_pdf_links(all_pdf_links)

def save_pdf_links(pdf_links):
    with open('pdf_links.txt', 'a') as file:
        file.write('\n'.join(pdf_links) + '\n')
    print('PDF links saved.')

scrape_documentation()
