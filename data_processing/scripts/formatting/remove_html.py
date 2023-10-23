from bs4 import BeautifulSoup

def remove_html_tags(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract text content, removing all tags
    text_content = soup.get_text(separator=' ')
    
    # Write cleaned content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text_content)

# Ask the user for a file path
file_path = input("Please enter the path to your file: ")

# Use the function
remove_html_tags(file_path)
