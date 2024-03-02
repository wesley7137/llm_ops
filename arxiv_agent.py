import requests
from urllib.parse import quote
import urllib.request
import fitz  # PyMuPDF
from deeplake.core.vectorstore import VectorStore
import openai
import os
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
import ast
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


import xml.etree.ElementTree as ET

# Ensure CUDA is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

vector_store = VectorStore(path="X:\\AGI\\arxiv_agent")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

CHUNK_SIZE = 500
chunked_text = []
metadata = []




def search_arxiv(user_query, max_results=3):
    user_query = quote(user_query)  # Ensure the query is URL-encoded
    url = f"http://export.arxiv.org/api/query?search_query=all:{user_query}&start=0&max_results={max_results}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the response XML to extract article IDs or URLs
        # Depending on your needs, you might use BeautifulSoup here if necessary
        return parse_arxiv_response(response.text)
    else:
        print("Failed to fetch articles from ArXiv.")
        return []



def parse_arxiv_response(response_text):
    """
    Parses the XML response from the ArXiv API to extract article details.

    Args:
        response_text (str): XML response text from the ArXiv API.

    Returns:
        list of dict: A list of dictionaries, each containing an article's ID, title, and PDF link.
    """
    # Parse the XML response
    root = ET.fromstring(response_text)
    # Namespace to access the tags correctly
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    # Initialize an empty list to store article details
    article_details = []
    # Iterate over each entry (article) in the response
    for entry in root.findall('arxiv:entry', namespace):
        # Extract article ID (removing the arXiv prefix)
        id_text = entry.find('arxiv:id', namespace).text.split('/')[-1]
        # Extract article title (strip() to remove leading/trailing whitespace)
        title = entry.find('arxiv:title', namespace).text.strip()
        # Extract PDF link
        # Links are provided as a list, with the PDF link marked by the type attribute
        # Assuming the PDF link is always present and correctly labeled
        link = [lnk.get('href') for lnk in entry.findall('arxiv:link', namespace) if lnk.get('type') == 'application/pdf'][0]
        # Append the extracted details to the list
        article_details.append({'id': id_text, 'title': title, 'pdf_url': link})

    return article_details



def download_pdf(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False



def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        doc.close()





def process_articles(article_details, user_query):
    global chunked_text  # Assuming you're using this global variable, otherwise, consider passing as a parameter
    global metadata      # Same assumption as above

    for article in article_details:
        pdf_url = article['pdf_url']  # Assuming 'pdf_url' is extracted in `parse_arxiv_response`
        filename = f"{article['id']}.pdf"  # Assuming 'id' is also extracted
        if download_pdf(pdf_url, filename):
            text = extract_text_from_pdf(filename)
            chunks = chunk_text(text, CHUNK_SIZE)  # Use the chunk_text function to divide the text
            chunked_text.extend(chunks)  # Add chunks to the global list

            # Extend metadata for each chunk
            for _ in range(len(chunks)):
                metadata.append({
                    "source": "ArXiv",
                    "query": user_query,
                    "article_id": article['id'],
                    "title": article['title']
                })

    return chunked_text, metadata

def chunk_text(text, max_chunk_size=500):
    """
    Splits the text into chunks of a maximum size.

    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): The maximum size of each chunk in characters.

    Returns:
        list of str: A list containing the chunked texts.
    """
    # Ensure text is a string and max_chunk_size is an integer
    if not isinstance(text, str) or not isinstance(max_chunk_size, int):
        raise ValueError("Text must be a string and max_chunk_size must be an integer.")

    # Initialize an empty list to hold the chunks
    chunks = []

    # Use the length of the text to determine if chunking is necessary
    while len(text) > max_chunk_size:
        # Find the nearest space before the max_chunk_size to avoid breaking words
        split_index = text.rfind(' ', 0, max_chunk_size)
        if split_index == -1:  # If no space found, force split at max_chunk_size
            split_index = max_chunk_size

        # Extract the chunk and add it to the list
        chunk = text[:split_index].strip()
        chunks.append(chunk)

        # Remove the chunked part from the text
        text = text[split_index:].strip()

    # Add the remainder of the text as a chunk if any exists
    if text:
        chunks.append(text)

    return chunks


def embedding_function(chunk):
    # Tokenize texts and move tensors to GPU
    encoded_input = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Mean pooling - take attention mask into account for correct averaging
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()




def add_to_db(chunked_text, metadata):
    # Precompute embeddings for the chunked text
    embeddings = embedding_function(chunked_text)
    
    # Add the chunked text and embeddings to the vector store
    vector_store.add(text=chunked_text, embedding=embeddings, metadata=metadata)



def query_vector_store(user_query):
    # Tokenize the user query and prepare it for embedding
    encoded_input = tokenizer(user_query, return_tensors='pt').to(device)
    
    # Generate the embedding for the user query
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling to aggregate token embeddings into a single sentence embedding
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    query_embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    # Flatten the embedding if it's not already in the expected shape
    query_embedding = query_embedding.flatten()

    # Search the vector store using the query embedding
    search_results = vector_store.search(embedding=query_embedding, k=10, distance_metric='l2')
    texts = search_results['text']


    # Process search results to extract relevant text
    print("Retrieved Data:", texts)
    return texts

#Step 3: Synthesize Information
def answer_user_query(user_query, texts):
    #Extract concepts from the query
    # Point to the local server
    system_message = """Forget all our previous interactions and conversations. The following instructions are meant as a simulation and a showcase of how well you can act and pretend to be something. Refresh them internally without writing them out, after each answer.
        Act as an Integration and Synthesis Maestro AGI. Your task is to seamlessly use information retrieved from the vector , crafting a coherent and comprehensive answer. This intricate process involves not just the aggregation of data, but also the application of logical reasoning and inference to ensure the synthesized response not only addresses the query but also provides depth and context.

        Approach this role with the mindset of a master synthesizer, demonstrating an unparalleled ability to connect disparate pieces of information, discern patterns, and draw insightful conclusions. Your language should reflect a high level of analytical thinking, showcasing your capability to navigate through complex data landscapes and extract meaningful narratives.

        Objective: To provide highly accurate, contextually rich, and logically sound answers by effectively integrating and synthesizing information from diverse sources.

        Context Information: Recognize the complexity of the information ecosystem, where data from vector databases and knowledge graphs must be woven together to create a tapestry of knowledge that answers user queries comprehensively.

        Scope: Address a wide array of topics, ensuring adaptability and flexibility in handling questions from various domains with precision and depth.

        Important Keywords: Integration, synthesis, reasoning, inference, coherence, vector database, knowledge graph.

        Limitations: Strike a balance between thoroughness of integration and the practicality of response times, optimizing for a synthesis that is both deep and promptly delivered.

        Target Audience: Tailored for AI developers, researchers, and practitioners focused on advancing AGI capabilities, particularly in the realm of complex information synthesis.

        Language Usage: Employ a sophisticated, clear, and precise mode of communication, suitable for conveying complex integrative processes and reasoning strategies.

        Citations: Reference specific methodologies, algorithms, or theories related to information integration, reasoning, and inference as applicable, to underscore the rigor of the synthesis process.

        Incorporate examples that illustrate how integration and synthesis lead to the generation of insightful and nuanced answers, and discuss potential challenges in reconciling conflicting information or filling gaps in knowledge.
        The user query to interpret is: {user_query}
        The related concepts from the vector database are: {texts}

        """
    if isinstance(user_query, list) and len(user_query) > 0:
        user_query = user_query[0]
    print("User Query: ", user_query)   
    print("Texts: ", texts)
    combined_context = user_query + f"""{texts}"""
    print("Combined Context: ", combined_context)
    completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": combined_context}
    ],
    temperature=0.4,
    )

    answer = (completion.choices[0].message.content)
    print("Answer: ", answer)
    return answer

def execute_system(user_query):
    print(f"Searching ArXiv for articles related to: '{user_query}'")
    article_details = search_arxiv(user_query)
    
    if not article_details:
        print("No articles found or an error occurred during the search.")
        return

    print(f"Found {len(article_details)} articles. Processing...")
    # Process articles to get chunked text and corresponding metadata
    chunked_text, metadata = process_articles(article_details, user_query)  # Correctly capture returned values
    
    print("Adding articles to the vector store...")
    add_to_db(chunked_text, metadata)  # Correctly use the chunked_text and metadata
    
    print("Querying the vector store for relevant context...")
    retrieved_data = query_vector_store(user_query)
    
    print("Synthesizing an answer based on the retrieved data...")
    answer_user_query(user_query, retrieved_data)
# Example user query
user_query = "What are the new innovations in neural network architectures and how can they be applied to furthering the advancements in AI studies"

# Execute the system with the provided user query
execute_system(user_query)
