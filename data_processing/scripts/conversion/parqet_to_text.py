import pandas as pd

# read the parquet file
data = pd.read_parquet("C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\articles_dataset.pt")

# write to a txt file
data.to_csv('articles_output.txt', index=False)
