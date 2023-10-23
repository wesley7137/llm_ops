import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# specify the delimiter that separates articles in your file
delimiter = "--- Start of Document"

# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# read the text file and split it into articles
with open("C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research.txt", "r", encoding='latin-1') as file:
    text = file.read()
articles = text.split(delimiter)

# Append "--- Start of Document" back to each split string except the last one
articles = [article + delimiter for article in articles[:-1]] + [articles[-1]]

# tokenize the articles
encodings = tokenizer(articles, truncation=True, padding=True)

# save the tokenized articles to a PyTorch dataset file
dataset = MyDataset(encodings)
torch.save(dataset, "articles_dataset.pt")
