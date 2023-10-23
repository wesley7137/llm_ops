import pandas as pd

def convert_txt_to_csv(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        try:
            data.append(line.strip())
        except ValueError as e:
            print(f"ValueError: {str(e)} with line: {line}")

    df = pd.DataFrame(data, columns=["text"])
    df.to_csv(output_file_path, index=False, encoding="utf-8", escapechar="\\")

convert_txt_to_csv("C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research.txt", "C:\\Users\\wesla\\New_AI\\Data\\processed\\quantum_research.csv")
