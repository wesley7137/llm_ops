import pandas as pd

# Load the data from a JSON file
df = pd.read_json('C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\processed\\part2.json')

# Write the data to a CSV file
df.to_csv('output.csv', index=False)
