


#%% Imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification, AutoTokenizer
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
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# Define color constants
USER_INPUT_COLOR = Fore.BLUE + Style.BRIGHT
AI_EXTERNAL_DIALOGUE_COLOR = Fore.GREEN + Style.BRIGHT
AI_INTERNAL_DIALOGUE_COLOR = Fore.YELLOW + Style.DIM
SYSTEM_PROCESS_COLOR = Fore.WHITE + Style.DIM

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
MAX_NODES = 28
MAX_EDGES = 100
from awq_inference import LLM  # Replace 'your_module' with the actual name of your module file



# Use the generate_response method to get a response
#user_prompt = "Tell me about how AI can be utilized in biotechnology and medical research to further the study of consciousness and the human mind."
#response = llm.generate_response(user_prompt)
#print(response)



def initialize_components_and_load_checkpoints():
    global bert_tokenizer, bert_model, sentiment_model, sentiment_tokenizer, llm
    global memory_store, knowledge_graph_gnn, logic_classifier
    # Initialize the LLM
    llm = LLM()
    # Initialize components
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    sentiment_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
    sentiment_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", padding=True)

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
    gnn_interpretation = (f"GNN Interpretation: The graph analysis shows a {semantic_label} with a {influence_score}.")
    return gnn_interpretation


def interpret_logic_classification(logic_probabilities):
    # Interpret logic probabilities
    logic_probabilities = logic_probabilities.cpu().squeeze()
    positive_count = sum(1 for prob in logic_probabilities if prob[1] > prob[0])
    negative_count = len(logic_probabilities) - positive_count

    if positive_count > negative_count:
        return "Overall Positive Outcome"
    elif positive_count < negative_count:
        return "Overall Negative Outcome"
    else:
        return "Neutral Outcome"




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
        print(f"Searching for memories similar to: {user_input} with a threshold of {relevance_threshold}")
        search_results = self.vector_store.search(
            embedding_data=user_input, 
            embedding_function=self.bert_embedding_function
        )
        print(f"Raw search results: {search_results}")  # Debugging print

        if all(isinstance(result, dict) and 'score' in result for result in search_results):
            relevant_results = [result for result in search_results if result['score'] >= relevance_threshold]
            print(f"Relevant search results: {relevant_results}")  # Debugging print
        else:
            print(f"{Fore.RED}{Style.DIM}Warning: Search results are not in the expected format." + Style.RESET_ALL)
            relevant_results = []

        return [result['text'] for result in relevant_results if 'text' in result]

def process_retrieved_memories(memory_embeddings):
    concatenated_memory_texts = ' '.join(memory_embeddings)
    return concatenated_memory_texts


def summarize_memories(objective, memory_texts, llm):
    # Generate the summarization prompt
    prompt = f"""
    Write a summary of the following text for {objective}. The text consists of memory embeddings related to the user's query. Only summarize the relevant info and try to keep as much factual information intact:
    "{memory_texts}"
    SUMMARY:
    """

    # Call the generate_response method correctly
    summary_output = llm.generate_response(prompt)
    if summary_output:
        summary_text = summary_output.split("SUMMARY:")[1].strip()
        return summary_text
    else:
        return "Unable to generate summary."





def generate_system_prompt(sentiment_label, gnn_interpretation, logic_analysis):
    # Format the analysis into a structured prompt
    prompt_analysis = f"Sentiment Analysis: {sentiment_label}, GNN Analysis: {gnn_interpretation}, Logic Analysis: {logic_analysis}"
    return prompt_analysis

def generate_external_dialogue(system_prompt, user_input):
    # Combine system prompt and user input for model processing
    combined_prompt = f"System Analysis: {system_prompt}\nUser Query: {user_input}"
    # Model processing (LLM) to generate response based on the combined prompt
    response = llm.generate_response(combined_prompt)  # Simplified call to the model's response generation
    return response

def generate_internal_dialogue(sentiment_label, gnn_interpretation, logic_classification_summary, memory_embeddings):
    # Placeholder for generating the internal dialogue
    internal_dialogue = f"Sentiment: {sentiment_label}, GNN: {gnn_interpretation}, Logic: {logic_classification_summary}, Memory: {memory_embeddings}"
    return internal_dialogue

def print_colored(text, color):
    """Print text in the terminal with specified color."""
    print(color + text + Style.RESET_ALL)

def generate_response(combined_context, user_input, sentiment_text, gnn_interpretation, logic_classification_interpretation, llm):
    analysis_text = f"Sentiment: {sentiment_text}, GNN: {gnn_interpretation}, Logic: {logic_classification_interpretation}"
    prompt_text = f"{analysis_text}\nUser Prompt: {user_input}"

    # Directly use the generate_response from LLM class
    generated_response = llm.generate_response(prompt_text)
    return generated_response

def generate_ai_dialogue(internal_dialogue, user_input, llm):
    prompt_text = f"{internal_dialogue}\nUser Query: {user_input}"
    
    # Directly use the generate_response from LLM class
    external_dialogue = llm.generate_response(prompt_text)
    
    # Printing external dialogue in green bright color
    print_colored("External Dialogue: " + external_dialogue, AI_EXTERNAL_DIALOGUE_COLOR)
    return external_dialogue

def main():
    # Initialize components and load checkpoints
    initialize_components_and_load_checkpoints()
    
    llm = LLM()  # Assuming LLM class is already defined and includes generate_response method
    
    # Hard-coded user input
    user_input = "Do you know any cute dog names for italian greyhound boys??"


    # BERT Encoding - Corrected Tokenizer Usage
    tokenized_input = bert_tokenizer.encode_plus(
        text=user_input,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    # BERT Model Processing
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
    print(SYSTEM_PROCESS_COLOR + "Performing Logic Classification..." + Style.RESET_ALL)
    logic_classification = logic_classifier(gnn_output)
    logic_probabilities = torch.softmax(logic_classification, dim=1)
    logic_classification_interpretation = interpret_logic_classification(logic_probabilities)
    print(SYSTEM_PROCESS_COLOR + f"Logic Classification Summary: {logic_classification_interpretation}" + Style.RESET_ALL)

    # Sentiment Analysis
    print(SYSTEM_PROCESS_COLOR + "Performing Sentiment Analysis..." + Style.RESET_ALL)
    sentiment_label = perform_sentiment_analysis(user_input, _CLASS_NAMES)

    # Memory Retrieval
    print(SYSTEM_PROCESS_COLOR + "Retrieving Similar Memories..." + Style.RESET_ALL)
    memory_embeddings = memory_store.query_similar_memories(user_input)

    # Flatten tensors to 2D
    print(SYSTEM_PROCESS_COLOR + "Flattening Tensors..." + Style.RESET_ALL)
    user_input_embeddings_flattened = user_input_embeddings_normalized.view(1, -1)
    gnn_context_flattened = gnn_output.view(1, -1)
    logic_probabilities_flattened = logic_probabilities.view(1, -1)
    memory_embeddings_flattened = torch.cat([normalize(torch.tensor(mem)).view(1, -1) for mem in memory_embeddings], dim=1) if memory_embeddings else None


    # Combine contexts
    print(SYSTEM_PROCESS_COLOR + "Combining Contexts..." + Style.RESET_ALL)
    combined_context = torch.cat([user_input_embeddings_flattened, gnn_context_flattened, memory_embeddings_flattened, logic_probabilities_flattened], dim=1) if memory_embeddings_flattened is not None else torch.cat([user_input_embeddings_flattened, gnn_context_flattened, logic_probabilities_flattened], dim=1)
    # Memory Retrieval and Processing
    # Memory Retrieval and Summary
    print(SYSTEM_PROCESS_COLOR + "Retrieving Similar Memories..." + Style.RESET_ALL)
    memory_embeddings = memory_store.query_similar_memories(user_input)
    processed_memories = process_retrieved_memories(memory_embeddings)
    memory_summary = summarize_memories("user query relevance", processed_memories, llm)
    print(SYSTEM_PROCESS_COLOR + "Memory Summary: " + memory_summary + Style.RESET_ALL)

    # Generate Internal Dialogue including memory details

    # Print user input once in blue
    print_colored(f"HUMAN: {user_input}", USER_INPUT_COLOR)

    # Generate Internal Dialogue
    internal_dialogue = f"(Thought: I know the following about the user input: Sentiment: {sentiment_label}, GNN: {gnn_interpretation}, Logic: {logic_classification_interpretation}, Memory: {memory_embeddings}. END OF ANALYSIS)"
    
    # Generate AI's External Dialogue
    external_dialogue = generate_ai_dialogue(internal_dialogue, user_input, llm)

    # Saving to memory
    print_colored("Saving to Memory...", SYSTEM_PROCESS_COLOR)
    memory_store.store_memory(user_input + " " + external_dialogue, {"source": "source_information", "response": external_dialogue})

# Run the main function
if __name__ == "__main__":
    main()
