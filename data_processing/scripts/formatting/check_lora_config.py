
from transformers import AutoModelForCausalLM

model_name = "c:\\Users\\wesla\\bci\\bci_scripts\\llm_train\\llama2_7b_chat_uncensored"

model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)
