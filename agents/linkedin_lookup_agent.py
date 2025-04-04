import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langsmith import Client

from tools.tools import get_profile_url_tavily


def lookup_linkedin_profile(profile_name: str) -> str:
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model="phi4-mini",
        temperature=0,
    )
    template = """
    You are a professional LinkedIn researcher.

    Your task is to find the **exact LinkedIn profile URL** of the person named: "{name_of_person}".

    🛑 Instructions:
    - Respond ONLY with the final answer in the format:
    Final Answer: https://www.linkedin.com/in/username
    - Do NOT include any explanation, introduction, or extra text.
    - Do NOT return company or group links—only personal profile links.
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedIn Page",
            description="Useful when you want to get the LinkedIn page url.",
            func=get_profile_url_tavily,
        )
    ]

    # Create a LANGSMITH_API_KEY in Settings > API Keys
    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
    react_prompt = client.pull_prompt("hwchase17/react", include_model=True)

    react_agent = create_react_agent(
        llm=llm, tools=tools_for_agent, prompt=react_prompt
    )
    agent_executor = AgentExecutor(
        agent=react_agent,
        verbose=True,
        metadata={"why": "it matters"},
        tools=tools_for_agent,
    )

    result = agent_executor.invoke(
        input={
            "input": PromptTemplate.format_prompt(
                self=react_prompt,
                tools=tools_for_agent,
                tool_names="ToolA, ToolB",
                input=profile_name,
                agent_scratchpad="",
            )
        },
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url


if __name__ == "__main__":
    profile_url = lookup_linkedin_profile("Eden Marco")
    print(profile_url)
