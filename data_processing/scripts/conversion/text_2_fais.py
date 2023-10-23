import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import nltk
from transformers import RobertaModel, RobertaTokenizer

nltk.download('punkt')

def load_documents(text_file_path):
    with open(text_file_path, 'r', encoding='ISO-8859-1') as file:
        text = file.read()
    return sent_tokenize(text)

def create_embeddings(documents):
    model = RobertaModel.from_pretrained('distilroberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    print("Creating embeddings...")
    embeddings = []
    for doc in tqdm(documents, total=len(documents)):
        inputs = tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings

def create_faiss_index(embeddings):
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, pkl_file_path):
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(index, f)

def main():
    text_file_path = "C:\\Users\\wesla\\arixbcipdfs.txt"
    pkl_file_path = "c:\\Users\\wesla\\faiss_index1.pkl"
    
    documents = load_documents(text_file_path)
    embeddings = create_embeddings(documents)
    
    index = create_faiss_index(embeddings)
    
    save_faiss_index(index, pkl_file_path)

if __name__ == "__main__":
    main()
