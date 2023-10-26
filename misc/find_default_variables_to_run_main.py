# Importing defaultdict from collections to properly initialize call_relationships.
from collections import defaultdict

# Identifying the call relationships between functions again.
call_relationships = defaultdict(list)

# Loop through each line of code to find where functions are called.
for idx, line in enumerate(code_lines):
    for function_name in function_lines.keys():
        if function_name + '(' in line:
            # Identify which function this line belongs to, if any.
            calling_function = None
            for fn, line_num in function_lines.items():
                if idx > line_num:
                    calling_function = fn
                else:
                    break
            if calling_function:
                call_relationships[calling_function].append(function_name)

call_relationships
