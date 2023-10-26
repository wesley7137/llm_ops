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

print(call_relationships)
