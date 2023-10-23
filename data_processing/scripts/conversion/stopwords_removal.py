from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

def remove_stop_words_and_names(input_file_path, output_file_path):
    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Read the content from the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove unnecessary spacing
    text = re.sub(r'\s+', ' ', text)

    # Remove citations like (Tegmark, 2017) or (Tegmark, 2017; Russell, 2019)
    text = re.sub(r'\(([A-Za-z\s]+,\s\d{4}(;\s[A-Za-z\s]+,\s\d{4})*?)\)', '', text)

    # Remove patterns like --- End of Generative Agents <text value> ---
    text = re.sub(r'--- End of Generative Agents .*?---', '', text)

    # Remove patterns like --- Start of Document <text value> ---
    text = re.sub(r'--- Start of Document .*?---', '', text)

    # Split the text by lines
    lines = text.split('\n')

    # Remove lines that only contain 2 to 3 capitalized words
    lines = [line for line in lines if not re.fullmatch(r'\b[A-Z][a-z]*\b(?:\s\b[A-Z][a-z]*\b){1,2}', line)]

    # Tokenize the lines by words and filter out stop words
    filtered_words = [word for line in lines for word in word_tokenize(line) if word.lower() not in stop_words]

    # Write the processed text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(" ".join(filtered_words))

    print(f"Stop words, names, citations, and document markers removed. New text file written to {output_file_path}")

# Path to the input text file
input_file_path = "D:\\PROJECTS\\Chat-with-Github-Repo\\src\\utils\\pdfdirectory2deeplake.txt"

# Path to the output text file (stop words removed)
output_file_path = "D:\\PROJECTS\\Chat-with-Github-Repo\\src\\utils\\pdfdirectory2deeplake_stopwords_removed.txt"

# Process the text file
remove_stop_words_and_names(input_file_path, output_file_path)
