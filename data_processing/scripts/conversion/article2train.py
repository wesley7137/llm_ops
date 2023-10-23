import os
import pandas as pd

# Define the directory where your text files are
directory = "C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research"

# Initialize lists to hold file content and file names
title_list = []
summary_list = []
contents_list = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Check the file is a .txt file
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:  # Open the file with explicit encoding specification
            lines = f.readlines()  # Read the file line by line
            title_list.append(lines[0].strip())  # Append the title to title_list
            summary_list.append(lines[1].strip())  # Append the summary to summary_list
            contents_list.append(' '.join(lines[2:]).strip())  # Append the contents to contents_list

# Create a DataFrame from the lists
df = pd.DataFrame(list(zip(title_list, summary_list, contents_list)), columns=['title', 'summary', 'contents'])

# Save the DataFrame to a .csv file
df.to_csv('training_data.csv', index=False)
