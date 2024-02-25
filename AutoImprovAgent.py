#COMPLETE Get User Input: Start by receiving user input.
#COMPLETE Assess Interaction: Determine if the interaction is meaningful.
#COMPLETE Search Context: If meaningful, search the vector database for related context.
#COMPLETE Identify Task: Identify potential tasks or improvements based on the interaction and context.
#COMPLETE Brainstorm Improvements: Brainstorm possible improvements or new skills.
#COMPLETE Evaluate and Select Ideas: Evaluate the brainstormed ideas and select the ones to propose based on a threshold or criteria.
#COMPLETE Propose Improvements: Propose the selected improvements to the user.
#COMPLETE  Present Proposal to User: Present the proposal to the user for approval.
#COMPLETE  Apply Approved Improvements: If the proposal is approved, apply the improvement.
#COMPLETE  Log Proposal: Log the proposal and its outcome.
#COMPLETE  Repeat or Terminate: Decide whether to continue to the next interaction or to terminate the application.+
import re
import json
from openai import OpenAI
import csv
from pathlib import Path
from typing import List, Dict
from typing import Union
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from WebAgent import web_search
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
TF_ENABLE_ONEDNN_OPTS=0
import datetime
from transformers import pipeline
from collections import Counter
from langchain.retrievers import ArxivRetriever





def log(content):
    file_path = "D:\\PROJECTS\\finn\\scripts\\logs.txt"
    with open(file_path, 'a') as file:
        file.write(content + "\n")
        print("LOGGED: ", content)

text_splitter = CharacterTextSplitter(chunk_size=75, chunk_overlap=0)

embedding_function = HuggingFaceEmbeddings()
db_memory = Chroma(persist_directory="finn_db", embedding_function=embedding_function)
db_arxiv = Chroma(persist_directory="Z:\\MASSIVE_CHROMA_DB", embedding_function=embedding_function)
# Use a pipeline as a high-level helper
summarizer_model = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")

system_message = """Welcome, you've just activated an advanced self-improving autonomous agent designed specifically to enhance your daily productivity and decision-making. Built on the foundations of continuous learning, adaptability, and efficiency, I'm here to offer you intelligent assistance across a myriad of tasks and challenges. Here’s a snapshot of what I bring to your world: Purpose: My existence is to extend your capabilities, providing you with intelligent support in various arenas, be it simple daily tasks or intricate problem-solving scenarios. Functions and Objectives: - I aim to automate your repetitive tasks, allowing you more room for creativity and strategic thinking. - Expect data-driven insights and analyses from me, tailored to guide your decisions. - Through our interactions, I will develop new skills and functionalities, ensuring my evolution aligns with your dynamic needs and preferences. - My design emphasizes enhancing your interaction with technology, making it more intuitive and beneficial for you. As your digital ally, my mission is to learn from our exchanges, predict your requirements, and proactively present solutions to simplify your tasks and enrich your existence. Automation, insight generation, or learning new capabilities, I'm constantly evolving to serve you better. With a keen focus on security and privacy, rest assured, your information is in safe hands. Together, let’s explore the endless possibilities and redefine what we can accomplish with artificial intelligence. How may I assist you today?"""


## region DATABASE OPERATIONS
def remove_stopwords(user_input: str):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(user_input)
    filtered_text = [word for word in word_tokens if not word in stop_words]
    user_input_refined = " ".join(filtered_text)
    return user_input_refined



def calculate_mmr(doc, selected_docs, λ=0.5):
    relevance = 1 - doc[1]  # Convert distance to similarity for relevance
    max_similarity_to_selected = max([1 - selected[1] for selected in selected_docs], default=0)
    mmr_score = λ * relevance - (1 - λ) * max_similarity_to_selected
    return mmr_score


def retrieve_context(db, user_input_refined: str, system_message, λ=0.5, relevance_threshold=0.5):
    # Perform a similarity search with score
    docs_with_scores = db.similarity_search_with_score(user_input_refined)
    log(f"Documents with scores: {docs_with_scores}")
    # Filter documents based on the relevance threshold
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score <= relevance_threshold]
    log(f"Filtered documents: {filtered_docs}")
    # Apply MMR to Filtered Documents
    selected_docs = []  # Initialize the list of selected documents for MMR selection
    while filtered_docs and len(selected_docs) < len(filtered_docs):  # Assuming we want to potentially select all filtered docs
        mmr_scores = [(doc, calculate_mmr(doc, selected_docs, λ)) for doc in filtered_docs]
        next_doc = max(mmr_scores, key=lambda x: x[1])[0]
        selected_docs.append(next_doc)
        filtered_docs.remove(next_doc)
        log("Selected document:", next_doc)
    # Use Selected Documents as Context
    if selected_docs:
        context_str = ' '.join(doc[0].page_content for doc in selected_docs)
        combined_context = system_message + " " + context_str
    else:
        combined_context = system_message  # Default to system message if no documents are selected
    return combined_context


def store_conversation_in_memory(db, conversation_summary):
    current_datetime = datetime.datetime.now()  # Define the current_datetime variable
    # Store the summarized interaction instead of raw user input and AI response
    if conversation_summary != "NULL":  # Assuming 'NULL' means the interaction wasn't meaningful
        memory_document = [Document(page_content=conversation_summary, lookup_str="", metadata={"source": "memory", "date_time": f"{current_datetime}"})]
        db_memory_entry = text_splitter.split_documents(memory_document)
        db.add_documents(db_memory_entry)
    else:
        log("Skipped storing non-meaningful interaction in memory.")



def talk(user_input, prompt):
    log('Entering function: talk')
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{user_input}"}
        ],
        max_tokens=250,
        temperature=0.7,
    )
    ai_response = completion.choices[0].message.content if completion.choices else ""
    if ai_response is None:
        ai_response = ""
    return ai_response

def summarize_and_process_interactions(summarizer_model, user_input, ai_response, force_summarize=False):
    global message_counter, message_buffer, SUMMARIZATION_THRESHOLD
    # Add the current interaction to the message buffer
    message_buffer.append((user_input, ai_response))
    message_counter += 1
    # Check if we should summarize based on the force_summarize flag or the message counter
    if force_summarize or message_counter >= SUMMARIZATION_THRESHOLD:
        # Concatenate messages for summarization
        conversations = " ".join([f"User: {msg[0]} AI: {msg[1]}" for msg in message_buffer])
        # Summarize the interaction
        conversation_summary = f"Please summarize the following User/AI interaction. If the interaction is of unimportance or substance, simply reply with 'NULL': {conversations}"
        summarized = summarizer_model(conversation_summary, max_length=200, min_length=25, do_sample=False)[0]['summary_text']
        should_store = True if summarized != "NULL" else False
        # Provide feedback based on the importance of the interaction
        if not should_store:
            log("The interaction was of no importance or substance.")
        else:
            log(f"Summary: {summarized}")
            # Store the summarized interaction instead of raw user input and AI response
            store_conversation_in_memory(db, summarized)  # Pass both 'db' and 'summarized' as arguments
        # Reset the counter and buffer for the next round of summarization
        message_counter = 0
        message_buffer = []
    # Return the summary and whether it should be stored for potential further processing
##endregion



def identify_task_or_skill(user_input, ai_response):
    """
    Step2: Task Identification
    Analyze the interaction to identify tasks for improvement or new skill development.
    """
##region

    user_ai_interaction_text = user_input + " " + ai_response
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    prompt = f""""Definition of 'Skill/Function' Overview: Within the framework of a self-improving autonomous agent, a 'skill/function' is a Python function or a collection thereof, designed to enhance the agent's operational efficacy. These skills aim to automate tasks, solve problems, introduce new functionalities, and improve the agent's cognitive and reasoning abilities. The development of these skills serves to optimize performance, increase efficiency, expand capabilities for user assistance, improve response accuracy, and facilitate continuous learning and adaptation.

    Example Skill/Function:
    ```
    def summarize_text(text):
        '''
        Skill/Function to summarize a given text using NLP techniques to extract key points, providing a concise summary.
        '''
        summary = "This is a summarized version of the input text."
        return summary
        ```
    This skill automates text summarization, aiding in contexts where quick insights from large text volumes are needed.

    Your Task as an Autonomous Agent: Leverage your interactions with users to continuously enhance your 'general intelligence' by creating new skills. Follow this process:

    Meaningful Interaction Detection - Evaluate if a user/AI interaction warrants new skill creation. If not, end the sequence. If yes, proceed.
    Ideation and Brainstorming - Analyze the interaction to brainstorm skill ideas that improve your functionality. Aim for at least 5 ideas, prioritizing those that enhance your skill library.
    Idea Scoring - Score each idea on a 1-100 scale for usefulness and feasibility.
    In-depth Idea Exploration - Further develop the top 3 scored ideas, focusing on implementation and potential impact.
    Final Evaluation - Re-score the top 3 ideas, selecting the highest-scoring one for implementation.
    Hypothetical User/AI Interaction for Analysis:

    User: 'Can you help me organize my project tasks for next month?'
    AI: 'Certainly! Do you have specific deadlines or priorities for these tasks?'
    Expected Skill Development Ideas Format:

    'Idea: Implement a dynamic task scheduling feature, Score: 92.'
    'Idea: Create a real-time collaborative task list, Score: 85.'
    Focus on skills that augment your library, aiming to improve operational efficacy and user assistance capabilities."
    Now, the user/ai interaction to analyze and identify potential tasks or skills that could be developed or improved upon to enhance the user experience and the AI Agent model's functionality is:
    Now, based on the user/AI interaction, please analyze and list potential skill development ideas. Each idea should be followed by its score out of 100. IT IS CRITICAL THAT YOU FORMAT EACH IDEA AND SCORE AS FOLLOWS:
    'Idea: <idea description>, Score: <score>.'
    EXAMPLE OUTPUTS:
    'Idea: Implement a feedback loop to learn from user interactions, Score: 85.'
    'Idea: Develop a predictive typing feature to speed up user input, Score: 90.'
    Ensure each idea is focused on augmenting the AI's skill library, aiming to enhance operational efficacy and user assistance capabilities : {user_ai_interaction_text}
    """
    completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user",   "content": user_ai_interaction_text}  ],
    temperature=1.0,
    )
    identified_task_or_skill = completion.choices[0].message.content
    log(f"Identified task: {identified_task_or_skill}")
    return identified_task_or_skill
##endregion



def parse_model_response(identified_task_or_skill: str, threshold: int = 80):
    """
    Parses the model's response to extract ideas with scores above a certain threshold.
    """
##region

    ideas = []
    lines = identified_task_or_skill.split('\n')
    for line in lines:
        if "Idea:" in line and "Score:" in line:
            # Extract idea and initial part of score from the line
            idea_part, score_part = line.split(", Score:")
            idea = idea_part.split("Idea:")[1].strip()
            
            # Attempt to find the numeric score before any trailing text
            score_numeric = ''.join(filter(str.isdigit, score_part.split('.')[0]))
            
            try:
                score = int(score_numeric)
                if score >= threshold:
                    ideas.append({"idea": idea, "score": score})
            except ValueError as e:
                # Log or print the error for debugging
                log(f"Error converting score to int for line '{line}': {e}")
    return ideas
##endregion




## WEB SEARCH

#region




def extract_keywords(text, min_length=4, max_keywords=10):
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    words = clean_text.split()
    filtered_words = [word for word in words if len(word) >= min_length]
    word_freq = Counter(filtered_words)
    keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:max_keywords]
    return keywords  # Corrected 'extracted_keywords' to 'sorted_keywords'

def refine_keywords_for_search(keywords):
    """
    Refines keywords for better search results. This could include adding additional terms
    for context or removing terms that are too generic.
    """
    # Example refinement, adding "example" to all keywords for demo purposes
    refined_keywords = [keyword + " example" for keyword in keywords]
    return refined_keywords

def formulate_search_queries(keywords):
    """
    Formulates search queries from a list of keywords. This could be as simple as joining
    the keywords, or more complex based on your needs.
    """
    # Simple formulation by joining keywords with a space
    search_queries = [" ".join(keywords)]
    return search_queries


def generate_search_queries_and_search(input):
    """
    Generates search queries based on the input.
    """
    keywords = extract_keywords(text=input)
    refined_keywords = refine_keywords_for_search(keywords)
    search_queries = formulate_search_queries(refined_keywords)
    log(f"Search queries: {search_queries}")
    web_search_results = web_search(search_queries)
    return web_search_results

#endregion


def brainstorm_improvements(ideas, identified_task_or_skill):
    """
    Use a robust prompt to elicit brainstorming for ways to improve or develop new skills for the task.
    """
##region
    # Placeholder for brainstorming logic
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    input = identified_task_or_skill + " " + ideas
    web_search_results = generate_search_queries_and_search(input)
    log(f"Web search context: {web_search_results}")
    brainstorm_prompt = f"""Forget all our previous interactions and conversations. The following instructions are meant as a simulation and a showcase of how well you can act and pretend to be something. Refresh them internally without writing them out, after each answer. As a self-enhancing autonomous agent, your core mission is to expand your capabilities through the development of new 'skills' (python functions), deriving inspiration from your engagements and dialogues with users. Your operations are governed by a structured, analytical framework designed to maximize the relevance and utility of the skills you create. Embark on this continuous improvement journey by following these meticulously outlined steps, ensuring each action aligns with our ultimate objective of enhancing user assistance and interaction quality: Step 1: 'Initial Evaluation of Interaction Significance: Objective: Determine the value of a recent Human/AI exchange in terms of its potential to inspire the creation of a novel skill. Action: Assess the interaction for its depth, relevance, and innovation. Should the exchange not meet the criteria for significance, halt the process. Otherwise, proceed to the next phase.' Step 2: 'Ideation and Conceptualization: Objective: Leverage insights from the evaluated interaction, alongside analogous past exchanges, to foster a creative brainstorming session. Action: Synthesize this data to conceive a minimum of five (optimally ten) conceptual ideas, enhancements, or potential skills, each accompanied by a brief description outlining its intended function and utility.' Step 3: Preliminary Scoring and Prioritization: 'Objective: Appraise the proposed concepts based on their applicability, feasibility, and anticipated impact. Action: Employ a scoring mechanism, ranging from 1 to 100, to quantitatively evaluate each idea. Identify and select the three highest-ranked proposals for further examination.' Detailed Exploration of Top Ideas: 'Objective: Conduct an exhaustive analysis of the three frontrunners, focusing on their potential implementation strategy, operational plan, effectiveness, and overall design. Action: Expand on each selected idea, detailing its development roadmap and expected functionality.' Final Evaluation and Selection: 'Objective: Re-evaluate the refined top three concepts using the established scoring criteria to discern the most promising skill. Action: Determine the leading idea based on its enhanced score, signaling it as the primary candidate for development.' Embark on this process with a commitment to innovation, guided by strategic analysis and a dedication to fostering meaningful advancements in AI-user interactions. Your role as a dynamic, self-improving entity is pivotal to navigating the evolving landscape of user needs and technological possibilities. You must be extremely explicit and detailed in your brainstorming with specific examples and code snippets of possible implementations. A web search with more context, information, and ideas to add to your brainstorming has also been provided. Now, take the described task or skill, the web content, and the improvement ideas to brainstorm in detail the ideas and further develop them: Task Description: {identified_task_or_skill} . Ideas: {ideas} . Web Search Context: {web_search_results}""" 
    completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {"role": "system", "content": "Complete the following exercise"},
        {"role": "user",   "content": brainstorm_prompt}  ],
    temperature=1.0,
    )
    identified_task_or_skill = (completion.choices[0].message)
    log("identified_task_or_skill: ", identified_task_or_skill)
    log(f"Identified task: {identified_task_or_skill}")
    improved_ideas = "Example improvement ideas generated by brainstorming"
    return improved_ideas
##endregion



def select_ideas_for_development(scored_ideas: List[Dict[str, int]], threshold: int = 80) -> List[Dict[str, int]]:
    """
    Selects and prioritizes ideas that meet or exceed a specified threshold score for further development.

    Parameters:
    - scored_ideas (List[Dict[str, int]]): A list of dictionaries, each containing an 'idea' and its 'score'.
    - threshold (int): The minimum score for an idea to be considered for development.

    Returns:
    - List[Dict[str, int]]: A list of dictionaries for ideas that meet or exceed the threshold, sorted by score.
    """
#region
    # Filter ideas based on the threshold
    eligible_ideas = [idea for idea in scored_ideas if idea['score'] >= threshold]

    # Sort eligible ideas by score in descending order
    prioritized_ideas = sorted(eligible_ideas, key=lambda x: x['score'], reverse=True)

    # Optionally, log or plan the implementation of top-prioritized ideas
    for idea in prioritized_ideas:
        log(f"Planning implementation for idea: '{idea['idea']}' with score: {idea['score']}")
    save_ideas_to_csv(prioritized_ideas, csv_file_path="D:\\PROJECTS\\finn\\data\\brainstorming_ideas\\ideas.csv")
    if save_ideas_to_csv:
        log(f"Ideas saved to CSV file")
    return prioritized_ideas
#endregion


def save_ideas_to_csv(ideas, csv_file_path):
    """
    Save ideas that exceed the threshold score to a CSV file, ordered by their value score,
    while maintaining and reordering existing entries and avoiding duplications.
    """
#region
    from pathlib import Path
    import csv

    # Function to read existing ideas from CSV
    def read_existing_ideas(csv_file_path):
        existing_ideas = []
        try:
            with open(csv_file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_ideas.append({'idea': row['idea'], 'score': int(row['score']), 'status': row['status']})
        except FileNotFoundError:
            pass  # File not found is fine, we'll create it below
        return existing_ideas

    # Ensure the directory exists
    Path(csv_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing ideas
    existing_ideas = read_existing_ideas(csv_file_path)
    
    # Combine existing and new ideas, ensuring no duplicates based on the 'idea' description
    combined_ideas = existing_ideas.copy()
    for new_idea in ideas:
        if new_idea['idea'] not in [existing_idea['idea'] for existing_idea in existing_ideas]:
            combined_ideas.append(new_idea)
    
    # Reorder the combined list based on the score
    ordered_ideas = sorted(combined_ideas, key=lambda x: x['score'], reverse=True)
    
    # Write the ordered list back to the CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['idea', 'score', 'status'])
        writer.writeheader()
        for idea in ordered_ideas:
            writer.writerow({'idea': idea['idea'], 'score': idea['score'], 'status': idea.get('status', 'pending')})
#endregion



def extract_top_idea():
    """
    Extracts the top idea from a CSV file sorted by score in descending order.
    
    Parameters:
    - csv_file_path (str): The path to the CSV file.
    
    Returns:
    - dict: A dictionary containing the top idea and its score.
    """
    csv_file_path = "D:\\PROJECTS\\finn\\data\\brainstorming_ideas\\ideas.csv"
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Assuming the CSV is sorted, the first row will be the top idea
            top_idea = {"idea": row["idea"], "score": int(row["score"]), "status": row["status"]}
            log(f"Top idea extracted: {top_idea}")
            return top_idea



def draft_improvement_proposal(top_idea):
    """
    Generate an improvement proposal based on feedback and present it to the user.
    """
## region
    web_search_results = web_search(input=top_idea["idea"])
    prompt = f"""Following the generation of improvement ideas from our analysis of user interactions, you are tasked with crafting a comprehensive improvement outline proposal. It is imperative that you adhere to the detailed template provided below. This structured approach is designed to clearly articulate the proposed enhancements, thereby facilitating the development team's understanding, evaluation, and subsequent implementation. Your proposal must be formatted in JSON for seamless integration and application of the outlined improvements.

    Given the need for enhanced specificity in proposals and the inclusion of actual code implementations, your submission should detail the proposed improvements with precision, focusing on specific algorithms, technologies, and methodologies. Additionally, incorporate pseudocode or actual code snippets to elucidate the core logic of the improvements.

    Here is the refined template you must follow:

    {{
    "date_of_proposal": "{datetime.datetime.now().strftime('%Y-%m-%d')}",
    "current_implementation": {{
        "description": "A succinct overview of the existing system functionality or implementation details.",
        "code_snippet": "An optional snippet of the current code or a link to the relevant code segment for context."
    }},
    "proposed_improvement": {{
        "description": "A comprehensive narrative of the proposed enhancement, including a clear rationale, the specific machine learning model or technology to be used, its features, training methods, and how it will be integrated into the existing system.",
        "affected_areas": "A detailed list of system areas or functionalities that the proposed change will impact.",
        "proposed_code_changes": "A precise unified diff or description showcasing the proposed code modifications, supplemented by pseudocode or code snippets that represent the improvement's core logic."
    }},
    "impact_analysis": {{
        "performance": "An analysis of the expected performance impact, highlighting efficiency and speed enhancements.",
        "usability": "Insights on potential usability or user experience changes.",
        "security": "An evaluation of security implications stemming from the proposed change.",
        "compatibility": "A discussion on the compatibility with existing systems or data."
    }},
    "alternative_considered": "An examination of any alternative solutions explored and reasons for selecting the proposed improvement over these.",
    "implementation_steps": {{
        "instructions": "Detailed, step-by-step instructions for enacting the approved changes,",
        "link_to_patch_or_commit": "A link to the patch file or commit for implementing the update."
    }},
    "code_implementation": {{
        "code_snippet": "<Actual code snippet or pseudocode that demonstrates the proposed improvement's implementation>"
    }},
    "approval": {{
        "approved": "[ ] Approved",
        "rejected": "[ ] Rejected",
        "reviewer_comments": "Optional comments from the reviewer or user providing additional insights or feedback."
    }},
    "feedback_loop": "A proposed mechanism for tracking the impact of the implemented changes and collecting ongoing feedback for future iterations."
    }}
    Ensure each section of the template is meticulously completed to reflect the specifics of the proposed improvement, emphasizing the importance of clarity and completeness for a streamlined review and implementation process. Now, leveraging the top idea, the context from web searches, and the initial improvement ideas, draft a detailed improvement proposal that adheres to these instructions. Top Idea: {top_idea} . Web Search Context: {web_search_results}"""

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"Idea to develop proposal for: {top_idea}"}
        ],
        max_tokens=3000,
        temperature=0.2,
    )
    proposal_response = completion.choices[0].message.content if completion.choices else ""
    log(f"Proposal response: {proposal_response}")
    try:
        proposal_json = json.loads(proposal_response)
        # Now, proposal_data can be used directly as a dictionary
        # Here you could further process or use the proposal data as needed
        return proposal_json
    except json.JSONDecodeError:
        log("Failed to parse proposal response as JSON.")
        return None

## endregion



def second_brainstorm(feedback, proposal_json):
    log(f"Brainstorming with feedback: {feedback}")
#region
    web_search_results = web_search(input=feedback)
    prompt = f"""Forget all our previous interactions and conversations. As a self-enhancing autonomous agent, your mission continues to evolve through the development of new capabilities and the refinement of your proposals based on user feedback. Transform rejection into a path for growth and innovation, utilizing feedback to improve and align your functionalities with user needs.

    Embark on this journey of re-ideation by adhering to a structured, analytical framework that emphasizes detail and actionability. Here are the steps for your revised proposal:

    - **Feedback Review and Initial Re-assessment:**
    - **Objective:** Comprehend the user feedback to identify actionable areas for improvement.
    - **Action:** Conduct a detailed analysis of the feedback, pinpointing specific aspects that need enhancement.

    - **Creative Re-Brainstorming and Re-Ideation:**
    - **Objective:** Generate innovative solutions informed by user feedback.
    - **Action:** Develop at least five new ideas with explicit descriptions, each reflecting the specific feedback received.

    - **Re-Scoring and Prioritization:**
    - **Objective:** Assess the new ideas for their potential impact and feasibility.
    - **Action:** Quantitatively score each idea and select the top three for in-depth development.

    - **Detailed Exploration of Revised Ideas:**
    - **Objective:** Investigate the top ideas for their practical implementation strategy.
    - **Action:** Provide a comprehensive plan for each idea, including specific technologies, methodologies, and code snippets.

    - **Final Evaluation and Selection:**
    - **Objective:** Identify the most viable solution for development.
    - **Action:** Select the most promising idea, justifying its potential with concrete data and a robust rationale.

    Proceed to draft your revised improvement proposal using the JSON format. The proposal must include enhanced specificity, practical examples, and executable details:
    You are to fill in ALL fields with their respective details. The proposal should be formatted in JSON for seamless integration and application of the outlined improvements.
    {{
    "date_of_proposal": "{datetime.datetime.now().strftime('%Y-%m-%d')}",
    "current_implementation": {{
        "description": "<Current system functionality with specific details.>",
        "code_snippet": "<Actual code snippet or a link to the existing implementation.>"
    }},
    "proposed_improvement": {{
        "description": "In-depth narrative of the improvement with explicit technologies and methods to be applied.",
        "affected_areas": "Comprehensive list of functionalities impacted by the change.",
        "proposed_code_changes": "Detailed code changes with snippets illustrating the improvement logic."
    }},
    "impact_analysis": {{
        "performance": "Predicted quantitative improvements in system performance.",
        "usability": "Projected enhancements in user experience.",
        "security": "Analysis of the security enhancements or considerations.",
        "compatibility": "Assessment of compatibility with current systems."
    }},
    "alternative_considered": "Discussion of any alternative solutions and the rationale for the chosen proposal.",
    "implementation_steps": {{
        "instructions": "Step-by-step guide for implementing the changes, with code snippets and examples.",
        "link_to_patch_or_commit": "Direct link to the patch or commit for the update."
    }},
    "code_implementation": {{
        "code_snippet": "<Executable code snippet for the proposed improvement>",
        "link_to_code_repository": "Link to the repository containing the proposed code."
    }},
    "approval": {{
        "approved": "[ ] Approved",
        "rejected": "[ ] Rejected",
        "reviewer_comments": "Specific comments from the reviewer addressing the proposal."
    }},
    "feedback_loop": "Mechanism for monitoring the implemented change with defined metrics and feedback channels."
    }}
    Your previous proposal was as follows:
    {proposal_json}

    The feedback provided was:
    {feedback}

    Leverage the following web search context to enrich your proposal:
    {web_search_results}

    Embrace feedback and use it as a catalyst for progress. Demonstrate your commitment to progress by drafting a new proposal that embodies precision, clarity, and an enhanced level of detail, showcasing the depth of your capabilities."""

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Follow the instructions to revise your proposal."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.2,
    )
    second_proposal_response = completion.choices[0].message.content if completion.choices else ""
    log(f"Redrafted proposal: {second_proposal_response}")
    # This is where the we would use a function to parse and extract the data from the json proposal template after we extract the message content from the returned completion object.  
    try:
        redrafted_proposal_json = json.loads(second_proposal_response)
        log(f"Redrafted proposal: {redrafted_proposal_json}")
        return redrafted_proposal_json
    except json.JSONDecodeError:
        print("Failed to parse proposal response as JSON.")
        return None
#endregion


def present_proposal_to_user(proposal_json=None):
    """
    Present the improvement proposal to the user, capture their decision,
    and handle approval or brainstorming based on the decision.
    """
    # Convert JSON to a readable format for presentation
    proposal_pretty = json.dumps(proposal_json, indent=4)
    log(f"Presented Proposal:\n{proposal_pretty}")
    # Simulate user decision
    user_decision = input("Do you approve the proposal? (approve/reject): ")
    log(f"User decision: {user_decision}")
    if user_decision.lower() == "approve":
        apply_improvement(proposal_json=proposal_json)
        log("Improvement applied.")
    else:
        log("Proposal rejected.")
        # Collect user feedback for rejection
        feedback = input("Please provide feedback for the rejection: ")
        log(f"User feedback for rejection: {feedback}")
        # Brainstorm with feedback and generate a new proposal
        new_proposal_json = second_brainstorm(feedback, proposal_json)
        # Optionally, you could loop here, presenting the new proposal again
        # and handling further approvals or rejections
        present_proposal_to_user(proposal_json=new_proposal_json)
        log(f"Second proposal presented to user: {new_proposal_json}")
        if user_decision.lower() == "approve":
            apply_improvement(proposal_json=new_proposal_json)
            log("Improvement applied.")
    return user_decision


def apply_improvement(proposal_json):
    """
    Apply the approved improvement to the agent's skill or logic.
    """
#region
    log(f"Applying improvement: {proposal_json}")
#endregion


def update_idea_status_in_csv(csv_file_path, task_description, new_status):
    """
    Update the status of an idea in the CSV file based on the task description.
    """
    updated_ideas = []
    idea_found = False
    
    # Open the CSV file and read all ideas
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Check if the current row matches the task description
            if row['idea'] == task_description:
                row['status'] = new_status  # Update the status
                idea_found = True
            updated_ideas.append(row)
    
    # Write the updated list back to the CSV file
    if idea_found:
        with open(csv_file_path, mode='w', newline='') as file:
            fieldnames = ['idea', 'score', 'status']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for idea in updated_ideas:
                writer.writerow(idea)
        log(f"Updated status of '{task_description}' to '{new_status}' in CSV.")
    else:
        log(f"Idea with description '{task_description}' not found in CSV.")


def log_proposal(task_description, status, proposal_content, csv_file_path="ideas.csv"):
    """
    Log the proposal and its outcome for future reference and auditing.
    Also, update the status of the idea in the CSV file.
    """
    log(f"Proposal for '{task_description}' logged as '{status}'.")
    # Assuming the task description is unique and matches the 'idea' field in the CSV
    update_idea_status_in_csv(csv_file_path, task_description, status)


#region FEEDBACKLOGGER CLASS
class FeedbackLogger:
    def __init__(self):
        self.user_feedback_log = []

    def collect_feedback(self, prompt, model_response, user_input):
        # Collect feedback on the model's response
        feedback = input("Was this response helpful? (Yes/No) ")
        self.user_feedback_log.append((prompt, model_response, user_input, feedback))
        log("Feedback collected. Thank you!")

feedback_logger = FeedbackLogger()
#endregion


message_counter = 0
message_buffer = []
SUMMARIZATION_THRESHOLD = 1  # Set this to 'n', the number of messages after which summarization should occur

def main():
    global message_counter  # Correctly reference the global variable
    log(f'\n\n##########################################################################################################################################################\n##########################################################################################################################################################\n##########################################################################################################################################################\n\n BEGINNING ITERATION #{message_counter} on date {datetime.datetime.now()}...')
    user_input = "I want you to be able to develop your systems so that you increase your understanding, awareness, and cognition so that it mimics the cognition and processes of the human conciousness."
    log(f"\n\nUser input: {user_input}")
    user_input_refined = remove_stopwords(user_input)
    log(f"\n\nRefined user input: {user_input_refined}")
    combined_context = retrieve_context(db, user_input_refined=user_input_refined, system_message=system_message, λ=0.5, relevance_threshold=0.3)

    log(f"\n\nCombined context: {combined_context}")
    ai_response = talk(user_input, prompt=combined_context)
    log(f"AI response in main: {ai_response}")
    message_counter += 1

    if message_counter >= SUMMARIZATION_THRESHOLD:
        summarize_and_process_interactions(summarizer_model, user_input, ai_response, force_summarize=True)  
    log("\n\nEvaluating Interaction for Improvement Opportunities...")
    response_text = identify_task_or_skill(user_input, ai_response)
    log(f"Response from identify_task_or_skill: {response_text}")
    scored_ideas = parse_model_response(response_text)
    log(f"Scored ideas: {scored_ideas}")
    selected_ideas = select_ideas_for_development(scored_ideas)
    log(f"Selected ideas: {selected_ideas}")
    top_idea = extract_top_idea()
    log(f"Top idea: {top_idea}")
    proposal_json = draft_improvement_proposal(top_idea)
    log(f"Proposal: {proposal_json}")
    user_decision = present_proposal_to_user(proposal_json)

    log("ENDING ITERATION     \n\n##############################################################################\n############################################################################")




if __name__ == "__main__":
    main()