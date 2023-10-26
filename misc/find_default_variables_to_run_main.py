from collections import defaultdict
from pprint import pprint

# Your existing code to populate call_relationships
# ...

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
