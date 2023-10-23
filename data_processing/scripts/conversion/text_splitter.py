def split_file():
    # Ask the user for the file path
    file_path = input("Please enter the path to the text document that you want to split: ")

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Set the size in MB
    size_in_mb = 5
    # Calculate size in bytes
    size_in_bytes = size_in_mb * 1024 * 1024

    # Initialize counter for file part names
    count = 0
    # Initialize variable to store data

    try:
        with open(file_path, 'r') as infile:
            data = infile.read(size_in_bytes)
            # While there is still data left
            while data:
                # Generate a part file
                part_file_name = f"{file_path}_part_{count}.txt"
                with open(part_file_name, 'w') as outfile:
                    # Write data to part file
                    outfile.write(data)
                # Increment the counter
                count += 1
                # Read the next chunk of data
                data = infile.read(size_in_bytes)
                print(f"Created part file {part_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("File split operation completed successfully.")

# Use the function
split_file()
