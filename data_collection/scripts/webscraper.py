import os
import time
import requests
from bs4 import BeautifulSoup
import json

class GitHubScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def get_soup(self, url):
        response = self.session.get(url)
        return BeautifulSoup(response.text, 'html.parser')

    def get_json_data(self, url):
        response = self.session.get(url)
        return response.json()

    def scrape_repository(self, repo_name):
        repo_url = f"{self.base_url}/{repo_name}"
        soup = self.get_soup(repo_url)
        file_links = soup.find_all('a', {'class': 'js-navigation-open link-gray-dark'})
        data = []
        for link in file_links:
            file_url = f"{self.base_url}{link['href']}"
            if file_url.endswith('.json'):
                json_data = self.get_json_data(file_url)
                data.append(json_data)
            time.sleep(1)  # To prevent rate limiting
        return data

if __name__ == "__main__":
    scraper = GitHubScraper("https://github.com")
    data = scraper.scrape_repository("your-repo-name")
    with open('output.json', 'w') as f:
        json.dump(data, f)
