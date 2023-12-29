import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.nn import Module
from torch_geometric.nn import GCNConv, global_mean_pool
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from deeplake.core.vectorstore import VectorStore
import torch

# Assuming the existence of a 'user_input' variable that contains the text input from the user.

# Define the LSTM network for emotional context
import torch
import torch.nn as nn
import torch.nn.functional as F
_CLASS_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment", 
    "disapproval", "disgust", "embarrassment", "excitement", "fear", 
    "gratitude", "grief", "joy", "love", "nervousness", 
    "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "neutral"
]

class KnowledgeGraphGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Initialize device

        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim).to(self.device)
        self.conv2 = GCNConv(hidden_dim, output_dim).to(self.device)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5).to(self.device)
        
    def forward(self, node_features, edge_index, batch_index):
        # Move tensors to the device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        if batch_index is not None:
            batch_index = batch_index.to(self.device)

    # Rest of your code...
        # First graph convolution layer with ReLU and dropout
        x = F.relu(self.conv1(node_features, edge_index))
        x = self.dropout(x)

        # Second graph convolution layer with ReLU
        x = F.relu(self.conv2(x, edge_index))

        # Pooling and aggregation
        if batch_index is not None:
            x = global_mean_pool(x, batch_index)
        else:
            x = torch.sum(x, dim=0, keepdim=True)

        return x



class MemoryStore:
    def __init__(self, vector_store_path='SUNNY_memory'):
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(path=self.vector_store_path)

    def store_memory(self, text, metadata):
        self.vector_store.add(text=[text], 
                              embedding_function=bert_embedding_function, 
                              embedding_data=[text], 
                              metadata=[metadata])

    def query_similar_memories(self, user_input):
        try:
            search_results = self.vector_store.search(embedding_data=user_input, 
                                                      embedding_function=bert_embedding_function)
            # Extract text from search results
            memories = []
            for result in search_results:
                if isinstance(result, dict) and 'text' in result:
                    memories.append(result['text'])
            return memories
        except ValueError as e:
            # Handle empty dataset or no relevant memories found
            print("No similar memories found. Error:", e)
            return []  # Return an empty list

            


# BERT Embedding Function
def bert_embedding_function(texts, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            # Get the average of the last hidden states to use as the embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().tolist())

    return embeddings


def tokenize_and_embed(text, tokenizer, embedding_model):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([token_ids])

    # Retrieve embeddings
    with torch.no_grad():
        embeddings = embedding_model(input_tensor)[0]
    return embeddings


def decode_emotional_context(output, emotion_decoder):
    # Decode the LSTM output into an emotional context
    logits = emotion_decoder(output)
    emotion_probs = F.softmax(logits, dim=1)
    emotion_label = torch.argmax(emotion_probs, dim=1)
    return emotion_label




def normalize(vector):
    norm = torch.norm(vector, p=2, dim=1, keepdim=True)
    return vector / norm



def sentence_to_graph(sentence, tokenizer, nlp_model):
    # Tokenize and encode the sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['attention_mask'].to('cuda')

    # Get BERT's last hidden state
    with torch.no_grad():
        outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
    nodes = outputs.last_hidden_state.squeeze(0).to('cuda')

    # Example of creating a simple chain graph
    num_tokens = input_ids.size(1)
    edge_index = []
    for i in range(num_tokens - 1):
        edge_index.append([i, i + 1])  # edge from token i to token i+1
        edge_index.append([i + 1, i])  # edge from token i+1 to token i (if undirected)

    # Convert to a tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to('cuda')

    return nodes, edge_index



def embed_nodes(nodes, embedding_model):
    # Get embeddings for each node
    with torch.no_grad():
        embeddings = embedding_model(nodes.unsqueeze(0))[0].squeeze(0)
    return embeddings


def build_edge_index(edges):
    return torch.tensor(edges, dtype=torch.long).t().contiguous()




# Define the main AI system
class AISystem(nn.Module):   
    _CLASS_NAMES = [
        "admiration", "amusement", "anger", "annoyance", "approval", 
        "caring", "confusion", "curiosity", "desire", "disappointment", 
        "disapproval", "disgust", "embarrassment", "excitement", "fear", 
        "gratitude", "grief", "joy", "love", "nervousness", 
        "optimism", "pride", "realization", "relief", "remorse", 
        "sadness", "surprise", "neutral"
    ]

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Initialize device

        # Initialize components with device allocation
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(self.device)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

        self.gnn = KnowledgeGraphGNN(768, 256, 128).to(self.device)
        self.memory_store = MemoryStore()
        # Initialize StableLM Zephyr 3B model and tokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(
            'stabilityai/stablelm-zephyr-3b',
            trust_remote_code=True,
            device_map="auto"
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')

    def generate_response(self, combined_context, user_input, sentiment_output, gnn_output_text):
        # Prepare prompt for StableLM Zephyr 3B model
        
        # Ensure combined_context is a string
        if not isinstance(combined_context, str):
            combined_context = ' '.join([str(item) for item in combined_context.cpu().numpy().tolist()])  # Convert to string if needed
        pre_prompt_text = f"After careful analysis, I have concluded that the sentiment detected is: {sentiment_output}, and the logical implications of the GNN analysis indicate: {gnn_output_text}. "
        prompt_text = f"User: {user_input}\nAI: {pre_prompt_text}"

        prompt = [{'role': 'user', 'content': prompt_text}]
        inputs = self.llm_tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )


        # Set pad token for the tokenizer if not set
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        # Generate response using StableLM Zephyr 3B
        tokens = self.llm.generate(
            inputs.to(self.llm.device),
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=True
        )

        # Decode and clean the response
        final_response = self.llm_tokenizer.decode(tokens[0], skip_special_tokens=False)
        return final_response

    # ... [rest of the AISystem class] ...


    def process_input(self, input_text, _CLASS_NAMES):
        tokenized_input = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        # BERT encoding
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_encoded = bert_output.last_hidden_state

        # Generate graph embeddings
        nodes, edge_index = sentence_to_graph(input_text, tokenizer=self.tokenizer, nlp_model=self.bert)
        nodes, edge_index = nodes.to(self.device), edge_index.to(self.device)
        gnn_output = self.gnn(nodes, edge_index, batch_index=None)

        # Process emotion and memory embeddings
        sentiment_output = self.sentiment_model(input_ids)[0]
        
        # Convert GNN output to text (this requires a specific implementation)
        gnn_output_text = self.convert_gnn_output_to_text(gnn_output)

        # Convert sentiment output to text (this requires a specific implementation)
        sentiment_text = self.perform_sentiment_analysis(user_input, _CLASS_NAMES)
        memory_embeddings = self.memory_store.query_similar_memories(user_input) # Assuming this method handles device correctly

        # Check if memory_embeddings is empty before combining contexts
        if memory_embeddings:
            combined_context = f"{input_text} {gnn_output_text} {sentiment_text} {memory_embeddings}"

        else:
            # Handle the case where memory_embeddings is empty
            combined_context = f"{input_text} {gnn_output_text} {sentiment_text}"
        # Generate response with Phi-2 model
        final_response = self.generate_response(combined_context, user_input, sentiment_text, gnn_output_text)

        # Convert combined_context to string if necessary
        if isinstance(combined_context, torch.Tensor):
            combined_context = combined_context.detach().cpu().numpy().tolist()
            combined_context = ' '.join([str(item) for item in combined_context])  # Convert list to string
        final_response = self.generate_response(combined_context, user_input, sentiment_text, gnn_output_text)

        combined_text = input_text + " " + final_response
        # Store new interaction in memory
        self.memory_store.store_memory(combined_text + " " + final_response, 
                                       {"source": "source_information", 
                                        "response": final_response})
        print("Memory stored:", combined_text + " " + final_response)   
        return final_response




    def perform_sentiment_analysis(self, input_text, _CLASS_NAMES):
        # Tokenize input for sentiment model
        inputs = self.sentiment_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Perform sentiment analysis
        with torch.no_grad():
            outputs = self.sentiment_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            sentiment_scores = torch.softmax(logits, dim=1)
            top_sentiment = torch.argmax(sentiment_scores, dim=1)

        # Convert top sentiment to text (using _CLASS_NAMES)
        sentiment_label = _CLASS_NAMES[top_sentiment.item()]
        return sentiment_label



    def convert_gnn_output_to_text(self, gnn_output):
        """
        Convert GNN output to a human-readable text format with semantic interpretation.
        """
        # Placeholder logic: Map GNN output to semantic labels and assign weights
        if gnn_output.mean() > 0.5:
            influence_score = "high influence"
            semantic_label = "positive trend"
        else:
            influence_score = "low influence"
            semantic_label = "negative trend"

        # Combine the semantic label and influence score into a coherent phrase
        gnn_interpretation = f"The graph analysis shows a {semantic_label} with a {influence_score}."
        return gnn_interpretation



    def combine_all_contexts(self, original_user_input, gnn_context, sentiment_output, memory_context=None):
        # Ensure original_user_input is a string
        if not isinstance(original_user_input, str):
            original_user_input = str(original_user_input)

        # Tokenize and encode user input using BERT and move to the correct device
        tokenized_input = self.tokenizer(original_user_input, return_tensors='pt', padding=True, truncation=True).to(self.device)
        bert_output = self.bert(**tokenized_input)
        user_input_embeddings = bert_output.last_hidden_state.mean(dim=1)

        # Normalize all context embeddings
        user_input_embeddings_normalized = normalize(user_input_embeddings)
        gnn_context_normalized = normalize(gnn_context)
        sentiment_output_normalized = normalize(sentiment_output)

        # Handle memory embeddings
        if memory_context:
            memory_embeddings = [normalize(self.bert(self.tokenizer(mem, return_tensors='pt').to(self.device))[0].mean(dim=1)) for mem, _ in memory_context]
            memory_embeddings = torch.cat(memory_embeddings, dim=0)
            combined_context = torch.cat([user_input_embeddings_normalized, gnn_context_normalized, sentiment_output_normalized, memory_embeddings], dim=1)
        else:
            # Combine contexts without memory embeddings
            combined_context = torch.cat([user_input_embeddings_normalized, gnn_context_normalized, sentiment_output_normalized], dim=1)

        return combined_context





# Example of how to use KnowledgeGraphGNN
gnn_input_dim = 768  # Example input dimension size
gnn_hidden_dim = 64  # Example hidden layer dimension size
gnn_output_dim = 128  # Example output dimension size

# Initialize the GNN
knowledge_graph_gnn = KnowledgeGraphGNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim)
memory_store = MemoryStore()

# Forward pass of the GNN



# Helper Functions
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def generate_synthetic_graph_data(num_nodes=10, num_edges=20, dim=768):
    node_features = torch.randn(num_nodes, dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    return node_features, edge_index

# Initialize and Test AISystem
ai_system = AISystem()
user_input = "I'm feeling a little lonely today. I'm single and getting older and I'm afraid I'll be alone for the rest of my life. "
original_user_input = user_input
input_text = user_input
# Generating synthetic graph data for demonstration
node_features, edge_index = generate_synthetic_graph_data()
batch_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Example batch index

# Forward pass of GNN with synthetic data
graph_embeddings = ai_system.gnn(node_features, edge_index, batch_index)
print("Graph Embeddings:", graph_embeddings)

response = ai_system.process_input(user_input, _CLASS_NAMES)
print("System Response:", response)

save_model(ai_system.gnn, 'knowledge_graph_gnn.pth')
