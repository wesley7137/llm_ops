import json

def validate_data_format(json_file_path):
    required_keys = ["topic", "task_description", "instructions", "context"]
    errors = []

    with open(json_file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                entry = json.loads(line)

                if not isinstance(entry, dict):
                    errors.append(f"Error on line {i}: Entry is not a dictionary.")

                else:
                    for key in required_keys:
                        if key not in entry:
                            errors.append(f"Error on line {i}: Entry does not contain key '{key}'.")
                        elif not isinstance(entry[key], str):
                            errors.append(f"Error on line {i}: Value of '{key}' is not a string.")

            except json.JSONDecodeError:
                errors.append(f"Error on line {i}: Invalid JSON syntax.")

    if errors:
        print("The following errors were encountered:")
        for error in errors:
            print(error)
    else:
        print("Data format is valid.")

    return not bool(errors)

# Validate the data format
validate_data_format('C:\Users\wesla\OneDrive\Desktop\bci\project_info\bioengineering.json')