import json
import csv
import re

def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as f:
        text = f.read()

    # Use regex to find all JSON objects in the text
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)

    # Convert each JSON object to a dictionary and add it to a list
    data = []
    for json_object in json_objects:
        try:
            data.append(json.loads(json_object))
        except json.JSONDecodeError:
            continue  # Ignore JSON objects that can't be decoded

    # Write the data to a CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['topic', 'instruction', 'output'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

json_to_csv('C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\processed\\part2.json', 'C:\\Users\\wesla\\OneDrive\\Desktop\\bci_new\\bci\\Data\\processed\\part3.csv')