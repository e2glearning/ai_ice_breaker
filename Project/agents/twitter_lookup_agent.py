import os
from re import M
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

    # template = """
    #    given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username
    #    In Your Final answer only the person's username"""

    template = """
    You are an intelligent assistant that can search for Twitter profiles.

    Given the full name: {name_of_person}

    Your task:
    - Find a link to the person's Twitter profile.
    - Extract the Twitter username from the profile URL.
    - Only return the username in your final answer (e.g., `elonmusk`).

    Follow this format:

    Thought: Do I need to use a tool? Yes  
    Action: TwitterSearch  
    Action Input: Find Twitter profile for {name_of_person}

    When finished, reply with:

    Final Answer: <only the person's Twitter username>
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Twitter Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
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

    twitter_username = result["output"]
    return twitter_username


if __name__ == "__main__":
    print(lookup(name="Elon Musk", model="openai", verbose=True))
