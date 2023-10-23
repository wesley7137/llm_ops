import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("C:\\Users\\wesla\\Downloads\\wizard-vicuna-7B-uncensored\\processed_ml_promptandcompletions.csv")

# Split the data into training and validation datasets
train, val = train_test_split(data, test_size=0.2)

# Save the datasets as CSV files
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
