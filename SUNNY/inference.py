

#%% Imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from deeplake.core.vectorstore import VectorStore
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style
_CLASS_NAMES = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", 
    "Caring", "Confusion", "Curiosity", "Desire", "Disappointment", 
    "Disapproval", "Disgust", "Embarrassment", "Excitement", "Fear", 
    "Gratitude", "Grief", "Joy", "Love", "Nervousness", 
    "Optimism", "Pride", "Realization", "Relief", "Remorse", 
    "Sadness", "Surprise", "Neutral"
]
# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Initialize device


# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
MAX_NODES = 28
MAX_EDGES = 100



def initialize_components_and_load_checkpoints():
    global bert_tokenizer, bert_model, sentiment_model, sentiment_tokenizer, llm, llm_tokenizer
    global memory_store, knowledge_graph_gnn, logic_classifier

    # Initialize components
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    sentiment_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
    sentiment_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", padding=True)

    llm = AutoModelForCausalLM.from_pretrained('stabilityai/stablelm-zephyr-3b', trust_remote_code=True, device_map="auto")
    llm_tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b', padding=True)

    memory_store = MemoryStore(user_input="user_input", vector_store_path='memory/SUNNY_memory')
    knowledge_graph_gnn = KnowledgeGraphGNN(768, 64, 128, None).to(device)
    logic_classifier = LogicClassifier(128).to(device)
    knowledge_graph_gnn.logic_classifier = logic_classifier

    # Load checkpoints
    load_checkpoint(knowledge_graph_gnn, '/root/SUNNY/model/checkpoint2-logic_cyclic/cyclic_gnn_checkpoint_epoch_49.pth')
    load_checkpoint(logic_classifier, '/root/SUNNY/model/checkpoint2-logic_cyclic/cyclic_classifier_checkpoint_epoch_49.pth')
    print("Components initialized and checkpoints loaded")

# Required imports and class definitions (as provided earlier)

def bert_embedding_function(texts, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().tolist())

    return embeddings


def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved at {filepath}")


def delete_previous_checkpoint(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted previous checkpoint: {filepath}")


def normalize(vector):
    norm = torch.norm(vector, p=2, dim=1, keepdim=True)
    return vector / norm


# Initialize GNN and LogicClassifier
def load_checkpoint(model, filepath):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Call .train() if further training is needed
    else:
        print(f"{Fore.WHITE}{Style.DIM}No checkpoint found at {filepath}" + Style.RESET_ALL)


def perform_sentiment_analysis(input_text, _CLASS_NAMES):
    
    # Tokenize input for sentiment model
    print(f"{Fore.GREEN}{Style.DIM}Performing sentiment analysis..." + Style.RESET_ALL)
    inputs = sentiment_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    # Perform sentiment analysis
    with torch.no_grad():
        outputs = sentiment_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        sentiment_scores = torch.softmax(logits, dim=1)
        top_sentiment = torch.argmax(sentiment_scores, dim=1)
    # Convert top sentiment to text (using _CLASS_NAMES)
    sentiment_label = _CLASS_NAMES[top_sentiment.item()]
    print(f"{Fore.GREEN}{Style.DIM}Sentiment Detected: {sentiment_label}" + Style.RESET_ALL)
    return sentiment_label



def convert_gnn_output_to_text(gnn_output):
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
    gnn_interpretation = (f"{Fore.GREEN}{Style.DIM}GNN Interpretation: The graph analysis shows a {semantic_label} with a {influence_score}." + Style.RESET_ALL)
    return gnn_interpretation


def interpret_logic_classification(logic_probabilities):
    # Interpret logic probabilities
    logic_probabilities = logic_probabilities.cpu().squeeze()
    interpretations = ["positive outcome" if prob[1] > prob[0] else "negative outcome" for prob in logic_probabilities]
    return interpretations[0] if len(interpretations) == 1 else interpretations




def pad_nodes(nodes, hidden_size):
    # Padding nodes tensor to have uniform size [1, MAX_NODES, hidden_size]
    current_size = nodes.size(1)
    if current_size < MAX_NODES:
        padding_size = MAX_NODES - current_size
        padding = torch.zeros((1, padding_size, hidden_size), dtype=nodes.dtype, device=nodes.device)
        nodes = torch.cat([nodes, padding], dim=1)
    # Ensure the tensor is correctly sized
    nodes = nodes[:, :MAX_NODES, :]  # Truncate to MAX_NODES if needed
    return nodes


def logical_structure_to_label(item):
    # True/False to 1/0 label conversion
    return 1 if item['blanks'] else 0


def sentence_to_graph(sentence, tokenizer, nlp_model, max_length=512):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
    nodes = outputs.last_hidden_state
    nodes = pad_nodes(nodes, bert_model.config.hidden_size)
    # Generate graph edges, ensuring that edge indices do not exceed MAX_NODES
    edge_pairs = []
    for i in range(MAX_NODES - 1):  # Generate edges for a linear chain for example purposes
        edge_pairs.append([i, i + 1])
        edge_pairs.append([i + 1, i])
    # Truncate the edges to ensure the number does not exceed MAX_EDGES
    edge_pairs = edge_pairs[:MAX_EDGES]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return nodes, edge_index





#%% Class definitions
class KnowledgeGraphGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, logic_classifier):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.logic_classifier = logic_classifier

    def forward(self, node_features, edge_index, batch_index):
        x = F.relu(self.conv1(node_features, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        if batch_index is not None:
            x = global_mean_pool(x, batch_index)
        else:
            x = torch.sum(x, dim=0, keepdim=True)
        return x


class LogicClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)
    

class MemoryStore:
    def __init__(self, user_input, vector_store_path='memory/SUNNY_memory', bert_embedding_function=bert_embedding_function):
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(path=self.vector_store_path)
        self.user_input = user_input
        self.bert_embedding_function = bert_embedding_function
        
    def store_memory(self, text, metadata):
        # Validate inputs
        if not text or not isinstance(metadata, dict):
            print(f"{Fore.RED}{Style.DIM}Invalid input for memory storage." + Style.RESET_ALL)
            return

        try:
            self.vector_store.add(
                text=[text], 
                embedding_function=self.bert_embedding_function, 
                embedding_data=[text], 
                metadata=[metadata]
            )
            print(f"{Fore.WHITE}{Style.DIM}Memory stored successfully." + Style.RESET_ALL)
        except Exception as e:
            print(f"{Fore.RED}{Style.DIM}Error storing memory: {e}" + Style.RESET_ALL)

    def query_similar_memories(self, user_input, relevance_threshold=0.7):
        search_results = self.vector_store.search(
            embedding_data=user_input, 
            embedding_function=self.bert_embedding_function
        )

        # Ensure that each item in search_results is a dictionary
        if all(isinstance(result, dict) and 'score' in result for result in search_results):
            relevant_results = [result for result in search_results if result['score'] >= relevance_threshold]
        else:
            # Handle the case where search_results are not in the expected format
            print(f"{Fore.RED}{Style.DIM}Warning: Search results are not in the expected format." + Style.RESET_ALL)
            relevant_results = []

        return [result['text'] for result in relevant_results if 'text' in result]


def generate_response(combined_context, user_input, sentiment_text, gnn_interpretation, logic_classification_interpretation, llm, llm_tokenizer):
    # Preparing the response prompt
    analysis_text = f"Analysis:\nSentiment: {sentiment_text}\nGNN: {gnn_interpretation}\nLogic: {logic_classification_interpretation}"
    user_prompt_text = f"User Prompt: {user_input}"
    prompt_text = f"{analysis_text}\n{user_prompt_text}"

    # Ensure combined_context is processed correctly
    if isinstance(combined_context, torch.Tensor):
        combined_context = combined_context.detach().cpu().numpy().tolist()
        combined_context = ' '.join([str(item) for item in combined_context])

    # Format the input for the LLM
    prompt = [{'role': 'user', 'content': prompt_text}]
    inputs = llm_tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt')

    # Generate tokens using the LLM
    tokens = llm.generate(inputs.to(llm.device), max_new_tokens=1024, temperature=0.6, do_sample=True)

    # Decode and clean the response
    generated_response = llm_tokenizer.decode(tokens[0], skip_special_tokens=True)
    return generated_response


def main():
    # Initialize components and load checkpoints
    initialize_components_and_load_checkpoints()

    # Hard-coded user input
    user_input = "I am feeling happy today."

    # BERT Encoding
    tokenized_input = bert_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    user_input_embeddings = bert_output.last_hidden_state.mean(dim=1)
    user_input_embeddings_normalized = normalize(user_input_embeddings)  # Normalize user input embeddings

    # Graph Generation and GNN Processing
    nodes, edge_index = sentence_to_graph(user_input, bert_tokenizer, bert_model)
    nodes, edge_index = nodes.to(device), edge_index.to(device)
    gnn_output = knowledge_graph_gnn(nodes, edge_index, batch_index=None)

    # Convert GNN output to text for interpretation
    gnn_interpretation = convert_gnn_output_to_text(gnn_output)

    # Logic Classification
    print("Performing Logic Classification...")
    logic_classification = logic_classifier(gnn_output)
    logic_probabilities = torch.softmax(logic_classification, dim=1)
    logic_classification_interpretation = interpret_logic_classification(logic_probabilities)

    # Sentiment Analysis
    print("Performing Sentiment Analysis...")
    sentiment_text = perform_sentiment_analysis(user_input, _CLASS_NAMES)

    # Memory Retrieval
    print("Retrieving Similar Memories...")
    memory_embeddings = memory_store.query_similar_memories(user_input)

    # Flatten tensors to 2D
    print("Flattening Tensors...")
    user_input_embeddings_flattened = user_input_embeddings_normalized.view(1, -1)
    gnn_context_flattened = gnn_output.view(1, -1)
    logic_probabilities_flattened = logic_probabilities.view(1, -1)
    
    # Flatten and concatenate memory_embeddings
    memory_embeddings_flattened = torch.cat([normalize(torch.tensor(mem)).view(1, -1) for mem in memory_embeddings], dim=1) if memory_embeddings else None

    # Combine contexts
    print("Combining Contexts...")
    if memory_embeddings_flattened is not None:
        combined_context = torch.cat([user_input_embeddings_flattened, gnn_context_flattened, memory_embeddings_flattened, logic_probabilities_flattened], dim=1)
    else:
        combined_context = torch.cat([user_input_embeddings_flattened, gnn_context_flattened, logic_probabilities_flattened], dim=1)

    # Generate External Dialogue
    print("Generating External Dialogue...")
    generated_response = generate_response(combined_context, user_input, sentiment_text, gnn_interpretation, logic_classification_interpretation, llm, llm_tokenizer)

    # Save to memory after generating response
    print("Saving to Memory...")
    combined_text = user_input + " " + generated_response
    memory_store.store_memory(combined_text, {"source": "source_information", "response": generated_response})

    print(f"Generated Response: {generated_response}")

if __name__ == "__main__":
    main()
