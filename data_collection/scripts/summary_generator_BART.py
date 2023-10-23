from transformers import BartForConditionalGeneration, BartTokenizer
import json

# Load BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Path to the JSON file containing the documents
json_file_path = "D:\\BCI\\pdfs_8_16_virtualbrainenv\\virtual_brain_scientific_articles_8_18.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    all_documents = json.load(json_file)

# Check if the JSON document is a list of dictionaries
if isinstance(all_documents, list) and all(isinstance(doc, dict) for doc in all_documents):
    for idx, document in enumerate(all_documents):
        try:
            article_contents = document["article_contents"]
            # Tokenize the article contents
            inputs = tokenizer([article_contents], max_length=1024, return_tensors='pt', truncation=True)
            # Generate summary
            summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=400, max_length=600, early_stopping=True)
            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Update the "summary" key with the generated summary
            document["summary"] = summary
            print(f"Processed document at index {idx}")
        except Exception as e:
            print(f"Error processing document at index {idx}: {e}")

    # Save the updated documents to a new JSON file
    output_filename = "summarized_documents.json"
    with open(output_filename, 'w') as json_file:
        json.dump(all_documents, json_file)

    print("Processing complete!")
else:
    print("The JSON document is not in the expected format.")
