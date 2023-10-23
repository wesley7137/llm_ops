import os
import json
from bs4 import BeautifulSoup

def scrape_html_files(directory):
    result = []
    
    # Iterate through all HTML files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            filepath = os.path.join(directory, filename)
            
            # Open and parse the HTML file
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                
                # Extract the title of the webpage
                title = soup.title.string if soup.title else "No Title"
                
                # Extract the contents of the webpage (excluding HTML tags)
                contents = ' '.join(soup.stripped_strings)
                
                # Append the title and contents to the result
                result.append({
                    "Title of Webpage": title,
                    "Webpage Contents": contents
                })
    
    return result

def main():
    directory = "c:\\Users\\wesla\\meth_support_system\\docs\\thevirtualbrain1\\docs.thevirtualbrain.org\\manuals\\UserGuide" # Replace with the path to your directory
    data = scrape_html_files(directory)
    
    # Write the result to a JSON file
    with open('TVBmanual.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
