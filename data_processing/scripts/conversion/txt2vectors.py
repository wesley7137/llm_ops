import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data
with open("C:\\Users\\wesla\\Downloads\\wizard-vicuna-7B-uncensored\\Data\\processed\\deepmindrepositories.txt", 'r', encoding='utf-8') as f:
    data = f.read().split('\n\n')

# Parse the data into a DataFrame
df = pd.DataFrame(data, columns=['content'])

# Add an ID column
df['id'] = range(1, len(df) + 1)

# Convert the 'content' column to vectors
df['vector'] = df['content'].apply(lambda x: model.encode(x))

# Prepare the data for upserting
ids = df['id'].tolist()
vectors = df['vector'].tolist()

# Upsert the vectors
pinecone.upsert(index_name="openai-embeddings-3c9cf7c", ids=ids, vectors=vectors)
