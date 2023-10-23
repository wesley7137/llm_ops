#pre-process text data for QLorA training with spacy

import spacy
import os

def preprocess_data(file_path):
    nlp = spacy.load('en_core_web_sm')

    with open(file_path, 'r') as file:
        text = file.read()

    # Parse the text with SpaCy
    doc = nlp(text)

    # Extract the sentences
    sentences = [sent.string.strip() for sent in doc.sents]

    # Join the sentences using the '<eos>' token
    preprocessed_text = '<eos>'.join(sentences)

    return preprocessed_text

def save_preprocessed_data(file_path, preprocessed_text):
    with open(file_path, 'w') as file:
        file.write(preprocessed_text)

def main():
    input_file_path = 'your_file_path_here.txt'
    output_file_path = 'preprocessed_file_path_here.txt'

    preprocessed_text = preprocess_data(input_file_path)
    save_preprocessed_data(output_file_path, preprocessed_text)

if __name__ == "__main__":
    main()
