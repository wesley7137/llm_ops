from collections import OrderedDict
import ast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Initialize a local Transformers model for NLU
model_path = "facebook/bart-large-cnn"  # Replace this with the path to your local model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def understand_directive(directive):
    inputs = tokenizer([directive], padding=True, truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output.lower().split(", ")

def decompose_functions(script):
    print("Decomposing functions...")
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"Function name: {node.name}")
            print("Arguments: ", [arg.arg for arg in node.args.args])
            print("Function body:")
            for stmt in node.body:
                print(ast.dump(stmt, annotate_fields=False))

def track_variables(script):
    print("Tracking variables...")
    tree = ast.parse(script)
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets[0]
            if isinstance(targets, ast.Name):
                variable_name = targets.id
                if variable_name not in assignments:
                    assignments[variable_name] = []
                assignments[variable_name].append(node.lineno)
    print("Variable Assignments:")
    for var, lines in assignments.items():
        print(f"{var} assigned at lines {lines}")

# Enhanced analyze_script function
def analyze_script(script):
    tree = ast.parse(script)
    summary = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_args = [arg.arg for arg in node.args.args]
            function_body = []
            important_vars = set()
            control_flow = []
            loops = []
            external_calls = []
            error_handling = []
            comments = []
            step_by_step = []  # List to hold step-by-step breakdown
            step_counter = 1  # Counter for step numbering

            for sub_node in ast.walk(node):
                # Variable Assignments and Usage
                if isinstance(sub_node, (ast.Assign, ast.Name, ast.AugAssign)):
                    targets = sub_node.targets if hasattr(sub_node, 'targets') else [sub_node]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            important_vars.add(target.id)
                            step_description = f"{step_counter}. Variable {target.id} is assigned at line {sub_node.lineno}"
                            step_by_step.append(step_description)
                            step_counter += 1

                # Function Calls
                elif isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    function_body.append(f"Calls function {sub_node.func.id}")
                    step_description = f"{step_counter}. Calls function {sub_node.func.id} at line {sub_node.lineno}"
                    step_by_step.append(step_description)
                    step_counter += 1

                # Control Flow
                elif isinstance(sub_node, ast.If):
                    control_flow.append(ast.dump(sub_node, annotate_fields=False))
                    step_description = f"{step_counter}. Conditional branch at line {sub_node.lineno}"
                    step_by_step.append(step_description)
                    step_counter += 1

                # Loops
                elif isinstance(sub_node, (ast.For, ast.While)):
                    loops.append(ast.dump(sub_node, annotate_fields=False))
                    step_description = f"{step_counter}. Loop starts at line {sub_node.lineno}"
                    step_by_step.append(step_description)
                    step_counter += 1

                # Exception Handling
                elif isinstance(sub_node, ast.Try):
                    error_handling.append(ast.dump(sub_node, annotate_fields=False))
                    step_description = f"{step_counter}. Exception handling starts at line {sub_node.lineno}"
                    step_by_step.append(step_description)
                    step_counter += 1

                # Comments and Documentation
                elif isinstance(sub_node, ast.Expr) and isinstance(sub_node.value, ast.Str):
                    comments.append(sub_node.value.s)

            summary[function_name] = {
                'Arguments': function_args,
                'Important Variables': list(important_vars),
                'Functionality': function_body,
                'Control Flow': control_flow,
                'Loops': loops,
                'External Calls': external_calls,
                'Error Handling': error_handling,
                'Comments': comments,
                'Step-by-Step Breakdown': step_by_step  # Include step-by-step breakdown in the summary
            }

    return summary


from collections import OrderedDict

def rename_functions(script_summary, rename_map):
    new_script_summary = OrderedDict()
    for old_name, details in script_summary.items():
        new_name = rename_map.get(old_name, old_name)
        new_script_summary[new_name] = details
    return new_script_summary

def rename_variables(script_summary, rename_map):
    for function, details in script_summary.items():
        new_vars = [rename_map.get(var, var) for var in details['Important Variables']]
        script_summary[function]['Important Variables'] = new_vars
    return script_summary

def change_order_of_functions(script_summary, new_order):
    new_script_summary = OrderedDict()
    for function in new_order:
        new_script_summary[function] = script_summary[function]
    return new_script_summary

def change_order_of_function_arguments(script_summary, function_name, new_order):
    script_summary[function_name]['Arguments'] = new_order
    return script_summary

def combine_functions(script_summary, combined_name, function_names):
    new_function = {
        'Arguments': [],
        'Important Variables': set(),
        'Functionality': []
    }
    for function_name in function_names:
        func_detail = script_summary[function_name]
        new_function['Arguments'].extend(func_detail['Arguments'])
        new_function['Important Variables'].update(set(func_detail['Important Variables']))
        new_function['Functionality'].extend(func_detail['Functionality'])
    new_function['Important Variables'] = list(new_function['Important Variables'])
    script_summary[combined_name] = new_function
    for function_name in function_names:
        del script_summary[function_name]
    return script_summary

def split_function(script_summary, original_function, new_functions):
    original_function_detail = script_summary[original_function]
    for new_function, details in new_functions.items():
        new_func_detail = {
            'Arguments': details.get('Arguments', []),
            'Important Variables': details.get('Important Variables', []),
            'Functionality': details.get('Functionality', []),
        }
        script_summary[new_function] = new_func_detail
    del script_summary[original_function]
    return script_summary


def generate_restructured_code(script_summary, user_requirements):
    print("Generating restructured code...")
    
    # Placeholder for the restructured code
    restructured_code = ""
    
    # Parse user requirements
    requirements = user_requirements.split(", ")
    
    # Make a deep copy of the script summary for manipulation
    new_summary = script_summary.copy()
    
    # Handle renaming of functions
    if 'rename functions' in requirements:
        for func in new_summary.keys():
            new_name = input(f"Enter new name for function {func} or press Enter to keep the old name: ").strip()
            if new_name:
                new_summary[new_name] = new_summary.pop(func)
    
    # Handle renaming of variables
    if 'rename variables' in requirements:
        for func, details in new_summary.items():
            for i, var in enumerate(details['Important Variables']):
                new_var_name = input(f"Enter new name for variable {var} in function {func} or press Enter to keep the old name: ").strip()
                if new_var_name:
                    details['Important Variables'][i] = new_var_name
    
    # Handle reordering of functions
    if 'reorder functions' in requirements:
        new_order = input("Enter the new order of functions separated by commas: ").split(", ")
        new_summary = {func: new_summary[func] for func in new_order}
    
    # Handle reordering of function arguments
    if 'reorder arguments' in requirements:
        for func, details in new_summary.items():
            new_arg_order = input(f"Enter the new order of arguments for function {func} separated by commas: ").split(", ")
            details['Arguments'] = new_arg_order
    
    # Handle combining of functions
    if 'combine functions' in requirements:
        functions_to_combine = input("Enter the names of functions to combine, separated by commas: ").split(", ")
        new_function_name = input("Enter the name of the new combined function: ").strip()
        combined_function = {
            'Arguments': [],
            'Important Variables': [],
            'Functionality': []
        }
        for func in functions_to_combine:
            if func in new_summary:
                combined_function['Arguments'].extend(new_summary[func]['Arguments'])
                combined_function['Important Variables'].extend(new_summary[func]['Important Variables'])
                combined_function['Functionality'].extend(new_summary[func]['Functionality'])
                del new_summary[func]
        new_summary[new_function_name] = combined_function
    
    # Handle splitting of functions
    if 'split functions' in requirements:
        function_to_split = input("Enter the name of the function to split: ").strip()
        new_function_names = input("Enter the names of the new functions after splitting, separated by commas: ").split(", ")
        if function_to_split in new_summary:
            original_function = new_summary[function_to_split]
            for new_func in new_function_names:
                new_summary[new_func] = original_function  # Placeholder: Actual logic may vary
            del new_summary[function_to_split]
    
    # Generate the restructured code based on the new_summary
    # Placeholder logic: In a real scenario, you'd generate Python code based on new_summary
    restructured_code = str(new_summary)
    
    return restructured_code



# Main loop for the interactive part
def main_loop():
    script_path = input("Enter the path of the Python script to reverse engineer: ")
    print('Debug: script_path:', script_path)
    user_directive = input("Enter your directive on what to do with the engineering: ")

    script_path = script_path.strip('"')
    with open(script_path, 'r') as f:
        script = f.read()
    
    tasks = understand_directive(user_directive)
    
    print('Debug: tasks:', tasks)  # Moved this line here after tasks has been defined
    
    if 'decompose functions' in tasks:
        decompose_functions(script)
    if 'track variables' in tasks:
        track_variables(script)
    
    script_summary = analyze_script(script)
    
    while True:
        print("Here is the summary of the existing script:")
        for function, details in script_summary.items():
            print(f"Function name: {function}")
            print(f"Arguments: {details['Arguments']}")
            print(f"Important Variables: {details['Important Variables']}")
            print(f"Functionality: {details['Functionality']}")
            print('-'*40)
        
        reconstruct = input("Would you like to reconstruct the script in a different way? (yes/no): ").strip().lower()
        if reconstruct == 'yes':
            user_requirements = input("Please specify your requirements for restructuring: ")
            print("Review of your requests:")
            print(user_requirements)
            print("Step-by-step plan for code reconstruction: TBD")
            confirm = input("Do you agree with the reconstruction plan? (yes/no): ").strip().lower()
            if confirm == 'yes':
                restructured_code = generate_restructured_code(script_summary, user_requirements)
                print("Restructured Code:")
                print(restructured_code)
        more_changes = input("Would you like to add any additional functionalities or make further alterations? (yes/no): ").strip().lower()
        if more_changes != 'yes':
            break

# Start the main loop
main_loop()