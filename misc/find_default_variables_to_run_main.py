from collections import defaultdict
from pprint import pprint

# Read the Python script into a list of lines
with open('pth/to/your/directory', 'r') as f:
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
# Function to print relationships in a hierarchical manner
def print_hierarchical_relationships(call_relationships, function_name, level=0, visited=None):
    if visited is None:
        visited = set()
    indent = '  ' * level
    print(f"{indent}- {function_name}")
    visited.add(function_name)
    
    for callee in call_relationships.get(function_name, []):
        if callee not in visited:
            print_hierarchical_relationships(call_relationships, callee, level + 1, visited)
        else:
            print(f"{indent}  - {callee} (circular dependency)")

# Pretty-print the call relationships
print("Pretty-Printed Function Call Relationships:")
pprint(call_relationships)
print()

# Print the hierarchical text format
print("Hierarchical Function Call Relationships:")
for root_function in call_relationships.keys():
    print_hierarchical_relationships(call_relationships, root_function)
