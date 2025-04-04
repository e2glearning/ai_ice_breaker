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

def lookup(name: str) -> str:
    # llm = ChatOpenAI(
    #     temperature=0,
    #     model_name="gpt-4o-mini",
    # )
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model="phi4-mini",
        temperature=0,
    )

    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""

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

    # react_agent = create_react_agent(
    #     llm=llm, tools=tools_for_agent, prompt=react_prompt
    # )

    # react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    print(lookup(name="Eden Marco Udemy"))