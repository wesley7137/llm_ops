import json

def convert_to_alpaca_format(file_name):
    data = []
    
    with open(file_name, 'r') as f:
        content = f.read()
        json_data = json.loads(content)

        for line_data in json_data:
            instruction = 'Provide an empathetic and supportive response'
            input_text = line_data['input']
            output_text = line_data['output']

            data_point = {
                'instruction': instruction,
                'input': input_text,
                'output': output_text
            }
            data.append(data_point)

    with open('addiction_counseling_synthetic_alpaca.json', 'w') as f:
        json.dump(data, f, indent=4)

convert_to_alpaca_format('C:\\Users\\wesla\\meth_support_system\\addiction_counseling_synthetic.json')
