import requests
from urllib.parse import quote
import urllib.request
import os
import torch
from openai import OpenAI
import xml.etree.ElementTree as ET
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata


client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Ensure CUDA is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model

CHUNK_SIZE = 500
chunked_text = []
metadata = []

model_name = "sentence-transformers/msmarco-MiniLM-L-12-v3"
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding function created.")

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
        print("URL: ", url)
        print(f"PDF downloaded successfully to {filename}.")
        
        return filename
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


import os
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        print("PDF opened successfully.")
        text = ""
        for page in doc:
            text += page.get_text()
        print("Text extracted from PDF.")

        # Define the text file path, here we use the same directory and name as the PDF file
        txt_path = os.path.splitext(pdf_path)[0] + '.txt'
        print(f"Text file path defined: {txt_path}")

        # Write the extracted text to the .txt file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Text written to file.")

        # Return the text file path
        return txt_path
    finally:
        doc.close()
        print("PDF file closed.")        
        

from langchain.document_loaders import PDFMinerLoader

from langchain_core.documents import Document

def process_articles(filename, embedding_function):
    print("CharacterTextSplitter imported.")
    # Load the extracted text into documents
    loader = PDFMinerLoader(file_path=filename)
    print("TextLoader initialized.")
    documents = loader.load_and_split()
    print("Documents loaded.", documents[0])

    filtered_docs = filter_complex_metadata(documents)
    print("Documents filtered.", filtered_docs[0])
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    print("TextSplitter initialized.")
    docs = text_splitter.transform_documents(filtered_docs)
    print("Documents split into chunks.", docs[0])
    # create the open-source embedding function
    embedding_function = embedding_function
    print("Embedding function created.")
    db = Chroma(persist_directory="Z:\\MASSIVE_CHROMA_DB")
    # Add documents to Chroma
    db.from_documents(docs, embedding_function, persist_directory="Z:\\MASSIVE_CHROMA_DB")
    print("Documents added to Chroma.")


def query_vector_store(user_query, embedding_function):
    print("Querying the vector store.")
    embedding_function = embedding_function
    db = Chroma(persist_directory="Z:\\MASSIVE_CHROMA_DB", embedding_function=embedding_function)
    retriever = db.as_retriever(search_type="similarity", k=5)
    print("Retriever initialized.")
    docs = retriever.get_relevant_documents(user_query)
    print("Documents retrieved from the vector store:", docs)  
    return docs


#Step 3: Synthesize Information
def answer_user_query(user_query, docs):
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
    print("Texts: ", docs)
    combined_context = user_query + f"""{docs}"""
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

def main():
    user_query = "How do I create and train a neural network using the Mamba LLM architecture? Generate a step by step guide with detailed and explicit directions and the relevant implemented code."
    # Step 1: Search for Information

    articles = search_arxiv(user_query)
    print("Articles found:", articles)
    # Step 2: Retrieve and Process Information
    for article in articles:
        filename = download_pdf(article['pdf_url'], f"{article['id']}.pdf")
        #txt_path = extract_text_from_pdf(f"{article['id']}.pdf")
        process_articles(filename, embedding_function)
    # Step 3: Synthesize Information
    docs = query_vector_store(user_query, embedding_function)
    answer = answer_user_query(user_query, docs)
    print("Answer:", answer)
    
if __name__ == "__main__":


    main()