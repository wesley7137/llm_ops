import os
import pandas as pd

# Specify the directory containing your text files
directory = ("C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research.txt")

# Read all text files and store their content in a list
articles = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r') as file:
            articles.append(file.read())

# Convert the list of articles to a DataFrame
df = pd.DataFrame(articles, columns=['text'])

# Save the DataFrame to a CSV file
df.to_csv("C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research.csv", index=False)
