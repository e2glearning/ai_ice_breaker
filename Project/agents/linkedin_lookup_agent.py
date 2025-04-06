import os
from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily
from langsmith import Client


def lookup(name: str, model: str, verbose: bool) -> str:
    openai_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )
    ollama_llm = ChatOllama(
        base_url="http://ollama:11434",
        model=os.environ["OLLAMA_MODEL"],
        temperature=0,
    )

    # template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page. choose the one you think matches most in case there are many from the list. Dont try to validate from external sources. trust yourself.
    #                           Your answer should contain only a URL"""

    template = """
    You are an intelligent assistant that can search for LinkedIn profiles.

    Given the full name: {name_of_person}

    Your task:
    - Find a LinkedIn profile link that most closely matches the given name.
    - If multiple profiles exist, choose the one that seems like the best match.
    - Do NOT validate using any external sources — trust your own judgment.

    Follow this format:

    Thought: Do I need to use a tool? Yes  
    Action: LinkedInSearch  
    Action Input: Find LinkedIn profile for {name_of_person}

    When finished, reply with:

    Final Answer: <only the URL of the LinkedIn profile>
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    # Create a LANGSMITH_API_KEY in Settings > API Keys
    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
    react_prompt = client.pull_prompt("hwchase17/react", include_model=True)

    if model == "ollama":
        llm = ollama_llm
    else:
        llm = openai_llm
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, verbose=verbose, handle_parsing_errors=True
    )

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    print(lookup(name="Eden Marco Udemy", model="openai", verbose=True))
