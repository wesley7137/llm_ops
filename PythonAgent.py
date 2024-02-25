def python_code_execute(input):
    from langchain.agents import create_react_agent
    from langchain import hub
    from langchain.agents import AgentExecutor
    from langchain_experimental.tools import PythonREPLTool
    from langchain_community.llms.openai import OpenAI
    from langchain.agents import Tool
    from langchain.tools import DuckDuckGoSearchRun

    tools = [PythonREPLTool()]
    tools += [
    Tool(
        name="Search",
        func=DuckDuckGoSearchRun().run,
        description="useful for when you need to search the web",
        )
    ]

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_react_agent(OpenAI(base_url="http://localhost:1234/v1", temperature=0), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        {
            "input": f"{input}",
        }
    )
