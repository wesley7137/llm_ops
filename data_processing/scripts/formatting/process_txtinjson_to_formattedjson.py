import csv

data = []

with open('C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\Raw\\hypothesis_dataset_processed_for_bert.txt', 'r') as f:
    for line in f:
        line = line.strip()  # Remove leading/trailing white spaces
        if line:  # Check if line is not empty
            topic, source, target = line.split(', ', 2)  # Split line into three parts
            # Remove leading/trailing quotes from each part
            topic = topic.strip('"')
            source = source.strip('"')
            target = target.strip('"')
            # Append to data
            data.append([topic, source, target])

# Write the data to a CSV file
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["topic", "source", "target"])  # Write header
    writer.writerows(data)  # Write data
