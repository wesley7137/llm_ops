from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAI
from typing import List
import os
import glob

# Define custom file management functions with detailed docstrings

def agent_read_file(filepath: str) -> str:
    """
    Reads and returns the contents of the specified file.

    Parameters:
    - filepath (str): Absolute or relative path to the file.

    Returns:
    - The contents of the file as a string.

    Example:
    >>> content = agent_read_file('path/to/myfile.txt')
    >>> print(content)
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def agent_write_file(filepath: str, contents: str) -> None:
    """
    Writes the given contents to the specified file. If the file doesn't exist, it's created.

    Parameters:
    - filepath (str): Absolute or relative path to the file.
    - contents (str): Content to write to the file.

    Returns:
    - None

    Example:
    >>> agent_write_file('path/to/myfile.txt', 'Hello, World!')
    """
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(contents)

def agent_search_files(directory: str, pattern: str) -> List[str]:
    """
    Searches for files matching the given pattern within the specified directory.

    Parameters:
    - directory (str): Directory to search in.
    - pattern (str): Pattern to match filenames against.

    Returns:
    - A list of paths to files matching the pattern.

    Example:
    >>> files = agent_search_files('/my/directory', '*.txt')
    >>> print(files)
    """
    return glob.glob(os.path.join(directory, pattern))

def agent_list_files(directory: str) -> List[str]:
    """
    Lists all files and directories in the specified directory.

    Parameters:
    - directory (str): Directory to list contents of.

    Returns:
    - A list of names of files and directories in the directory.

    Example:
    >>> items = agent_list_files('/my/directory')
    >>> print(items)
    """
    return os.listdir(directory)



def python_code_execute(input_code: str):
    # Initialize the Python REPL tool with custom file management functions in its environment
    python_repl_tool = PythonREPLTool()
    tools = [python_repl_tool]
    tools.append(
        Tool(
            name="List Files in Directory",
            func=agent_list_files,
            description="Lists all files and directories within a specified directory.",
        )
    )
    tools.append(
        Tool(
            name="Search Files in Directory",
            func=agent_search_files,
            description="Searches for files matching a specified pattern within a directory and returns a list of matching file paths.",
        )
    )
    tools.append(
        Tool(
            name="Read File",
            func=agent_read_file,
            description="Reads and returns the contents of the specified file.",
        )
    )
    tools.append(
        Tool(
            name="Write File",
            func=agent_write_file,
            description="Writes the given contents to the specified file. If the file doesn't exist, it's created.",
        )
    )
    
    from tempfile import mkdtemp
    working_directory = mkdtemp(dir="D:\\PROJECTS\\finn\\data\\working_directory")

    
    instructions = """You are an agent designed to write and execute python code to answer questions...
    """
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    
    agent = create_react_agent(OpenAI(base_url="http://localhost:1234/v1", temperature=0), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Invoke the agent with the input code
    result = agent_executor.invoke({"input": input_code})
    return result
python_code_execute({"input": "List the files located in the directory ' D:\\PROJECTS\\finn'."})