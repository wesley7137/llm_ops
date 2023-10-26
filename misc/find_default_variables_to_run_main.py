from collections import defaultdict
from pprint import pprint


# Read the Python script into a list of lines
with open('path/to/your/python_script.py', 'r') as f:
    code_lines = f.readlines()

# Initialize a dictionary to store function names and the line numbers where they are defined
function_lines = {}

# Loop through each line of code to find function definitions
for idx, line in enumerate(code_lines):
    line = line.strip()  # Remove leading and trailing whitespaces
    if line.startswith("def "):
        function_name = line.split('(')[0][4:]
        function_lines[function_name] = idx

# Your existing code starts here
from collections import defaultdict
call_relationships = defaultdict(list)

for idx, line in enumerate(code_lines):
    for function_name in function_lines.keys():
        if function_name + '(' in line:
            calling_function = None
            for fn, line_num in function_lines.items():
                if idx > line_num:
                    calling_function = fn
                else:
                    break
            if calling_function:
                call_relationships[calling_function].append(function_name)


# Function to print relationships in a hierarchical manner
def print_hierarchical_relationships(call_relationships, function_name, level=0):
    indent = '  ' * level
    print(f"{indent}- {function_name}")
    for callee in call_relationships.get(function_name, []):
        print_hierarchical_relationships(call_relationships, callee, level + 1)

# Pretty-print the call relationships
print("Pretty-Printed Function Call Relationships:")
pprint(call_relationships)
print()

# Print the hierarchical text format
print("Hierarchical Function Call Relationships:")
for root_function in call_relationships.keys():
    print_hierarchical_relationships(call_relationships, root_function)
This code snippet first uses pprint to display the call_relationships dictionary in a more readable format. After that, it uses the print_hierarchical_relationships function to display the relationships in a hierarchical manner.

Just insert this snippet at the point in your code where you want to inspect the call_relationships dictionary. This should make it easier to understand the relationships between functions in your code.







print(call_relationships)
