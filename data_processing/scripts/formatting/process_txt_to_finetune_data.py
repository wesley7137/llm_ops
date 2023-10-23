import nltk

# Download Punkt Sentence Tokenizer. This only needs to be done once.
nltk.download('punkt')

def convert_raw_text_to_finetune_format(input_file_path, output_file_path):
    # Open the input file and read the text.
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split the raw text into paragraphs (assumed to be separated by two newlines).
    paragraphs = raw_text.split('\n\n')

    # Open the output file.
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Process each paragraph.
        for paragraph in paragraphs:
            # Use NLTK's sent_tokenize function to split the paragraph into sentences.
            sentences = nltk.sent_tokenize(paragraph)
            # Write each sentence to the output file on a new line.
            for sentence in sentences:
                f.write(sentence + '\n')
            # Write an extra newline to separate documents.
            f.write('\n')

def main():
    # Ask the user for the input and output file paths.
    input_file_path = input("Enter the path to your input file: ")
    output_file_path = input("Enter the path to your output file: ")
    
    # Use the function.
    convert_raw_text_to_finetune_format(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
