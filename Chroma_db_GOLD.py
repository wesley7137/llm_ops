import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

client = chromadb.PersistentClient(path="test")
embedding_function = HuggingFaceEmbeddings()

# Create a list of dictionaries with 'page_content' and 'metadata' fields

document_contents = "This is documents content"

# Create Document objects instead of dictionaries
documents = [Document(page_content=document_contents, metadata={'some_value': "some_value"})]



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = Chroma.from_documents(docs, embedding_function, persist_directory="test")
