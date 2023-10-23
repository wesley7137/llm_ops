import json

# Open the JSON file
with open('file.json', 'r') as f:
    data = json.load(f)

# Open the text file
with open('file.txt', 'w') as f:
    for item in data:
        f.write("\n###Start_of_Article###\n")
        f.write(f"Topics: {item['topics']}\n")
        f.write(f"Summary: {item['summary']}\n")
        f.write(f"Article Contents: {item['article contents']}\n")
        f.write("###End_of_Article###\n")
