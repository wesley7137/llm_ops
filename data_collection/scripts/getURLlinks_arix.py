import urllib
import feedparser

def search_arxiv(query, max_results=10, file_path='links.txt'):
    base_url = 'http://export.arxiv.org/api/query?'
    query = urllib.parse.quote_plus(query)
    response = urllib.request.urlopen(base_url + f'search_query={query}&start=0&max_results={max_results}')
    feed = feedparser.parse(response)

    if not feed.entries:
        print('No results found')
        return

    try:
        # Load previously saved links
        with open(file_path, 'r') as f:
            previous_links = {line.strip() for line in f}
    except FileNotFoundError:
        previous_links = set()

    new_links = set()
    for entry in feed.entries:
        # Extract link to PDF
        for link in entry.links:
            if 'title' in link and link.title == 'pdf':
                new_links.add(link.href)
                break

    # Filter out previously saved links
    new_links = new_links - previous_links

    if new_links:
        with open(file_path, 'a') as f:  # Open the file in append mode
            for link in new_links:
                f.write(f'"{link}",\n')
        print(f'{len(new_links)} new results saved to {file_path}')
    else:
        print('No new results found')

# example usage
search_arxiv('computer')
