import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langsmith import Client


def lookup_linkedin_profile(profile_name: str) -> str:
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.2:3b-instruct-q8_0",
        temperature=0,
    )
    template = """
    Give a person name {name_of_person}, I want you to provide me with link to their proifle page on LinkedIn platform.
    The answer should only contain a link in valid url format "http(s)://<link-to-person-linkedin-profile-page>"
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedIn Page",
            description="Useful when you want to get the LinkedIn page url.",
            func="?",
        )
    ]

    # Create a LANGSMITH_API_KEY in Settings > API Keys
    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
    react_prompt = client.pull_prompt("hwchase17/react", include_model=True)
    react_agent = create_react_agent(
        llm=llm, tools=tools_for_agent, prompt=react_prompt
    )
    result = AgentExecutor(
        agent=react_agent,
        verbose=True,
        metadata={"why": "it matters"},
        tools=tools_for_agent,
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url
