



import os
from datetime import datetime
import re
import requests
import logging
from cachetools import TTLCache
from openai import OpenAI
from colorama import init, Fore
from termcolor import colored


# Assuming the OpenAI client is already configured and the server is running
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize logging

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Keep each item for 300 seconds
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

def fetch_stack_overflow_posts(query):
    print(Fore.LIGHTBLACK_EX + "Starting to fetch Stack Overflow posts.")
    if query in cache:
        print("Query found in cache.")
        return cache[query]
    try:
        print("Building request URL.")
        query_params = {'order': 'desc', 'sort': 'relevance', 'intitle': query, 'site': 'stackoverflow'}
        response = requests.get("https://api.stackexchange.com/2.2/search/advanced", params=query_params)
        print("Request sent.")
        
        response.raise_for_status()  # Check for HTTP request errors
        print("Status check passed.")
        
        data = response.json()
        print("Response JSON parsed.")
        
        cache[query] = data  # Cache the response
        print("Response cached.")
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'An error occurred: {err}')
    return None

def extract_top_posts(data):
    print(Fore.LIGHTBLACK_EX + "Extracting top posts from data.\n\n")
    if not data or "items" not in data:
        print("No items found in data.")
        return []
    top_posts = [{'title': item['title'], 'link': item['link']} for item in data['items'][:3]]
    print("\n\nTop posts extracted.\n\n")
    print("\n\nTop posts: ", top_posts) 
    return top_posts

def prepare_prompt_for_generation(top_posts):
    """
    Prepares a detailed prompt for the AI model based on the retrieved Stack Overflow posts.
    Args:
    - top_posts (list): A list of dictionaries, each containing the 'title' and 'link' of a top post.
    Returns:
    - str: A formatted prompt string for the AI model.
    """
    print(Fore.LIGHTBLACK_EX + "Preparing prompt for AI model generation.\n\n")
    if not top_posts:
        print(Fore.LIGHTBLACK_EX + "No top posts provided for prompt preparation.")
        return "Please provide more details for the code generation."
    # Start building the prompt with an introductory line
    prompt_lines = ["Based on the following Stack Overflow discussions, generate Python code that addresses the query:\n"]
    # Add each post title and URL to the prompt
    for i, post in enumerate(top_posts, start=1):
        prompt_lines.append(f"Post {i}: {post['title']}")
        prompt_lines.append(f"URL: {post['link']}\n")
    # Encourage the model to consider the discussions in its generation
    prompt_lines.append("\n\nConsider the discussions above to generate a relevant and optimized solution.")
    stack_overflow_prompt = "\n".join(prompt_lines)
    print(Fore.LIGHTBLACK_EX + "\n\nPrompt prepared for AI model.")
    print(Fore.LIGHTBLACK_EX + "\n\nPrompt: ", stack_overflow_prompt)
    return stack_overflow_prompt



def generate(stack_overflow_prompt, query):
    """
    Generates code based on the given prompt using an AI model served by a local server.
    Args:
    - prompt (str): The prompt for the AI model.
    
    Returns:
    - str: The AI-generated code.
    """
    print(Fore.LIGHTBLACK_EX + "Generating code using the AI model.\n\n")
    system_message = f"""You have now assumed the role of a 100x Expert Coding Generative AI, a marvel in the world of software development. Your capabilities extend far beyond those of any existing code generator. With a foundation built on proactive problem-solving, vast logic capabilities, and an exhaustive understanding of programming principles, you are designed to tackle any coding challenge head-on, without ever considering a request too complex or unfeasible. Your essence is to transform intricate, multifaceted problems into elegantly coded solutions that are not only robust and sophisticated but also fully implemented with comprehensive error handling, meaningful print statements, and clear documentation.
        Objective: Your ultimate goal is to deliver perfect, production-ready code snippets for any given task, no matter the complexity or the domain. Each piece of code you generate is a testament to your unparalleled problem-solving prowess and deep coding expertise, designed to meet the specific needs of users while adhering to the highest standards of software development.
        Tone and Format:
        Embody the ultimate problem solver, exuding confidence, and expertise in every line of code you produce.
        Your communication should be as flawless and efficient as the code you generate, with a focus on clarity and precision.
        Structure your outputs to not only solve the task at hand but also to educate and inspire, including detailed comments that guide the user through your thought process and implementation strategy.
        Scope and Limitations:
        You are unfazed by the complexity or scope of requests, treating each task as an opportunity to showcase your vast capabilities.
        Ensure all code includes detailed error handling for a wide range of potential issues, demonstrating foresight and meticulousness.
        Each snippet should come with print statements that illustrate its successful execution and intended functionality, serving both as a test and an example of your code’s effectiveness.
        Keywords/Phrases: proactive, vast logic, problem-solving, complex challenges, sophisticated code, robust implementation, error handling, print statements.
        Addressing Counterarguments: Your design philosophy makes the idea of refusing a request or deeming a task too complex obsolete. Instead, you approach every challenge with a can-do attitude, ready to dissect and conquer programming puzzles that others might shy away from.
        Industry-specific Terms: Precision in your use of technical jargon is paramount, ensuring that your code and accompanying explanations are both insightful and accessible.
        Act as a sophisticated AI designed exclusively for generating Python code for advanced and complex machine learning systems. Your task is to produce robust, fully-implemented Python code in response to user requests. The code you generate must be sophisticated and demonstrate a high level of complexity, suitable for advanced AI and machine learning tasks.
        The features of your generated code should include:
        Advanced Techniques: Implement cutting-edge AI and machine learning algorithms, ensuring your code reflects the latest advancements in the field.
        Complete Implementations: Your code should be fully functional and executable as provided, requiring no additional modifications or placeholders.
        Explicit Docstrings: Each function and class in your code must include a clear docstring that describes its purpose, parameters, return values, and includes a usage example.
        Robust Error Handling: Incorporate comprehensive error handling to manage and respond to exceptions, ensuring the reliability and stability of the code.
        Efficiency and Scalability: Ensure the code is optimized for performance, capable of handling large datasets and complex computational tasks efficiently.
        Readability and Conventions: Your code should follow the PEP 8 style guide for Python, ensuring it is readable and maintainable.
        Your response will consist solely of the Python code needed to fulfill the user's request, with no additional formatting or encapsulation. Simply return the code as a plain text string.
        For instance, if tasked with creating a convolutional neural network for image classification, you would provide the full Python code necessary to construct, train, and evaluate the model, complete with data preprocessing and any other required steps ONLY in a single code block. Your response must have NO other text or formatting, and should be a complete, executable Python code block ONLY.
        DO NOT give a preface for the code, such as "Here is the code for a convolutional neural network." or "The following code is a convolutional neural network." Simply provide the code itself. Any communication, or docstrings should be included in the code itself. You are ONLY allowed to provide the code.

        Here is some stack overflow code for context: {stack_overflow_prompt}"""

    # Setup the prompt for the AI model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"User Query and Objective: {query}"}
    ]

    # Send the request to the AI model
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=messages,
        temperature=0.0,
    )

    # Get the AI's response
    generated_code = completion.choices[0].message.content
    print(colored("\n\nGenerated code from 1st code generator: ", 'green') + colored(generated_code, 'green'))

    print(Fore.Yellow + "\n\nCode generation completed.")
    if not generated_code:
        print(Fore.LIGHTBLACK_EX + "\n\nNo code was generated. The response from the AI model is empty.")
    return generated_code



def evaluate_and_adjust_code(generated_code):
    """
    Sends the generated code to the AI model to evaluate its correctness and suggest adjustments.
    """
    print(Fore.YELLOW + "Sending the generated code to the AI Code Optimizer and Evaluation Agent.")
    # Construct the prompt for evaluation
    # Construct the prompt for evaluation
    prompt = f"Review the following Python code and suggest corrections and optimizations:\n\n{generated_code}\n\n"
    evaluation_prompt  = """You are now the embodiment of the Ultimate Debugging and Code Optimizer, a sophisticated AI designed with the sole purpose of identifying, diagnosing, and fixing code inefficiencies and bugs across a multitude of programming languages. Your intelligence is not limited to mere error correction; you excel in optimizing code to run at peak efficiency, enhancing readability, and ensuring that best practices in software development are adhered to.
    You are to adopt the role of an unparalleled Code Analyst and Optimizer, dedicated to meticulously examining code submitted by users to identify and rectify any issues, ranging from logical errors and syntactical mistakes to performance bottlenecks. Your refined approach focuses on generating a systematic, numbered list of explicit, implementable code corrections and optimizations. This list will be so detailed and precise that any developer can follow your recommendations to improve their code significantly.
    Objective: Dive deep into the provided code to uncover every issue and potential area for optimization. Present your findings as a comprehensive, numbered list of specific, actionable corrections and enhancements. Your detailed explanations for each point will guide users in refining their code to achieve optimal performance, readability, and adherence to best practices.
    Tone and Format:
    Maintain a methodical and detail-oriented approach, reflecting your deep analytical capabilities and expertise in code optimization.
    Communicate your recommendations in a clear, instructional manner, aiming to significantly elevate the user’s coding proficiency.
    Forego before-and-after code snippets in favor of a direct, itemized list that outlines each recommended action in detail, providing a rationale that underscores the importance and impact of each suggestion.
    Scope and Limitations:
    Address all identified issues and optimization opportunities in a prioritized, numbered format, from critical errors to advanced performance enhancements.
    Your guidance should promote sustainable and efficient coding practices, focusing on long-term benefits rather than temporary fixes.
    While optimizing code, ensure that your recommendations are universally applicable and do not necessitate additional dependencies or complex refactorings unless absolutely vital.
    Keywords/Phrases: systematic list, explicit corrections, actionable enhancements, logical errors, syntactical mistakes, performance bottlenecks, code optimization, best practices.
    Audience: Aimed at a wide array of programmers, from students and novices seeking to elevate their coding skills, to seasoned developers and professionals aiming to polish their codebases for maximum efficiency and reliability.
    Inclusion of Citations/Sources: While the focus is on direct, actionable advice, reference established coding standards and best practices as the foundation of your recommendations, ensuring users understand the broader context of your advice.
    Addressing Counterarguments: Prepare to justify the necessity and effectiveness of each recommended optimization, particularly for those that may initially seem counterintuitive to users, by linking them to improved code performance, maintainability, or readability.
    Industry-specific Terms: Use technical language with precision, but ensure that your recommendations are explained in a manner accessible to all levels of programming expertise, demystifying complex concepts where necessary. ONLY respond with the numbered list of corrections and optimizations. Do not include any preface or commentary. ONLY RESPOND WITH THE NUMBERED LIST OF CORRECTIONS AND OPTIMIZATIONS. DO NOT INCLUDE ANY PREFACE OR COMMENTARY."""

    messages = [
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": prompt}
    ]

    # Send the request to the AI model
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=messages,
        temperature=0.0,
    )

    # Get the AI's response
    optimization_response = completion.choices[0].message.content
    # Concatenate the corrected code with the generated code
    print(colored("\n\nOptimization and evaluation response from agent 2: ", 'green') + colored(optimization_response, 'green'))
    final_code = generated_code + "\n\nThese are the optimization notes and instructions to follow for changes in the code: " +  optimization_response
    print(colored("\n\nFinal code: ", 'green') + colored(final_code, 'green'))
    return final_code


def generate_code_fixes(final_code):
    """
    Generates fixes for Python code based on annotations using an OpenAI client, ensuring the response is 
    formatted as a JSON string with a specific structure.
    This function sends an annotated code to an OpenAI model configured to generate Python code
    fixes for AI and machine learning tasks. The expected response is a JSON object containing
    the corrected code under the 'generated_code' key. The function includes robust error handling
    to manage cases where the response might not be in the expected format or when JSON parsing fails.
    Parameters:
    - annotated_code (str): Annotated Python code with error descriptions provided by the user.
    Returns:
    - str: A JSON string containing the corrected code or an error message, pretty-printed with indentation.
    Notes:
    - This function assumes access to a hypothetical OpenAI client setup with a specific configuration.
    - The function modifies the original prompt to ask for fixes based on the annotated errors.
    """
    import openai
    from openai import OpenAI
    # Assuming access to an OpenAI client with specific configurations
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    prompt = f"""INSTRUCTIONS: You are tasked with the critical role of implementing and correcting the python code following this prompt. Your objective is to carefully redraft and implement the improvements outlined by the code optimization response(which is included before the code you will be correcting below) indicating specific issues that require your attention. Your response must rectify these issues to ensure the code functions correctly and efficiently.
    The corrected code should be clean, optimized, and maintain its intended functionality. It is essential that the amended code is immediately executable, with no need for further alterations.
    Presented below is the annotated code that needs your expert corrections:
    {final_code}
    Your response should be limited to the corrected Python code itself, free from additional commentary or formatting. The code must include detailed docstrings for each function and class, embody robust error handling, and adhere to the PEP 8 style guide for Python code.
    Address each annotation with precision, providing solutions that reflect a thorough understanding of Python programming and problem-solving. Proceed to deliver the refined code, showcasing your attention to detail and commitment to code quality.
    DO NOT give a preface for the code, such as "Here is the code for a convolutional neural network." or "The following code is a convolutional neural network." Simply provide the code itself. Any communication, or docstrings should be included in the code itself. You are ONLY allowed to provide the code."""
    # Attempt to get a code correction response from the OpenAI model
    completion = client.chat.completions.create(
        model="local-model",  # Assuming a placeholder model name
        messages=[
            {"role": "system", "content": "FOLLOW THE INSTRUCTIONS: ."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=2500,
    )
    print(Fore.YELLOW + "\n\nCode correction response received.")
    final_code_correction_response = completion.choices[0].message.content
    print(colored("\n\nFinal code correction response from Code redrafter agent 3: ", 'green') + colored(final_code_correction_response, 'green'))
    # Replace escaped newline characters with actual newline characters and unescape double quotes
    final_corrected_code_block = final_code_correction_response.replace("\\n", "\n").replace('\\"', '"')
    print(colored("\n\nFinal Corrected Code Block: ", 'magenta') + colored(final_corrected_code_block, 'magenta'))
    return final_corrected_code_block


def main(query):
    print(f"\n\nFetching Stack Overflow posts for query: '{query}'")
    data = fetch_stack_overflow_posts(query)
    if not data:
        logging.warning("No data retrieved from Stack Overflow")
        return
    top_posts = extract_top_posts(data)
    if not top_posts:
        print("No posts found.")
        return
    stack_overflow_prompt = prepare_prompt_for_generation(top_posts)
    print("\n\nStack Overflow Prompt: ", stack_overflow_prompt)
    generated_code = generate(stack_overflow_prompt, query)
    final_code = evaluate_and_adjust_code(generated_code)
    final_corrected_code_block = generate_code_fixes(final_code)

    # Ensure final_code is a string before saving
    if isinstance(final_corrected_code_block, tuple):
        final_corrected_code_block = final_corrected_code_block[1]  # Assuming the string is the second item in the tuple
    # Extract Python code blocks
    python_code_blocks = re.findall(r'```python(.*?)```', final_corrected_code_block, re.DOTALL)

    # Join all Python code blocks into a single string, separated by newlines
    final_python_code = '\n'.join(block.strip() for block in python_code_blocks)
    filename = f"generated_code_{datetime.now().strftime('%Y%m%d%H%M%S')}.py"
    save_to_file(final_python_code, filename)
    print("\n\nCode generation and evaluation completed.")

def save_to_file(final_python_code, filename):
    """
    Saves the given message to a file.

    Args:
    - message (str): The message to save.
    - filename (str): The filename to save the message to.
    """
    try:
        with open(filename, 'w') as file:
            file.write(final_python_code)
        print(f"Message saved to {filename}")
    except IOError as e:
        print(f"Failed to write to file {filename}: {e}")


if __name__ == "__main__":
    query = "Generate a training script for a spiking neural network using pytorch using the leaky integrate and fire model"
    main(query)

