
def web_search(input):
    from langchain.agents import Tool, AgentType, initialize_agent 
    from langchain_community.utilities import SerpAPIWrapper 
    from langchain_openai import OpenAI 
    import os

    os.environ["SERPAPI_API_KEY"] = "6cdae9e7d9a0a496d959ce02d732a0202ee47d1d4b134dfab286b4812f726078"
    params = {
        "engine": "google",
        "gl": "us",
        "hl": "en",
    }
    serp_api = SerpAPIWrapper(params=params)
    tools = [
    Tool(
        name="SerpAPI",
        func=serp_api.run,
        description="Useful for answering questions that require a search query.",
        ),
        ]
    llm = OpenAI()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    web_search_results = agent.invoke(input)
    print(web_search_results)
    return web_search_results

