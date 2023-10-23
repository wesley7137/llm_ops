import pandas as pd

# Load a CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\wesla\\New_AI\\Data\\processed\\train.csv")

# Save the DataFrame as a Parquet file
df.to_parquet('C:\\Users\\wesla\\New_AI\\Data\\processed\\train.parquet', index=False)  # Set index=False to drop the index
