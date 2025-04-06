from typing import Tuple
import os
import string
from tabnanny import verbose
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import re
from langchain_core.runnables import RunnableLambda
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets
from output_parsers import Summary, summary_parser


def ice_break_with(name: str, model: str, verbose: bool) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name, model=model, verbose=verbose)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )

    twitter_username = twitter_lookup_agent(name=name, model=model, verbose=verbose)
    tweets = scrape_user_tweets(username=twitter_username, mock=True)

    # summary_template = """
    # given the information about a person from linkedin {information},
    # and their latest twitter posts {twitter_posts} I want you to create:
    # 1. A short summary
    # 2. two interesting facts about them

    # Use both information from twitter and Linkedin
    # \n{format_instructions}
    # """

    summary_template = """
    You are an intelligent assistant that can analyze data from LinkedIn and Twitter to generate insights.

    Given the LinkedIn information: {information}  
    And the latest Twitter posts: {twitter_posts}  

    Follow this reasoning format step by step:

    Thought: Do I need to use a tool? Yes  
    Action: UseSummaryGenerator  
    Action Input: Generate a short summary and two interesting facts using the information provided above.

    {format_instructions}

    When you are done using tools, respond with:

    Final Answer: <your complete answer here>
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    openai_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )
    ollama_llm = ChatOllama(
        base_url="http://ollama:11434",
        model=os.environ["OLLAMA_MODEL"],
        temperature=0,
        extract_reasoning=False,
    )

    if model == "openai":
        llm = openai_llm
    else:
        llm = ollama_llm

    chain = summary_prompt_template | llm | summary_parser

    res: Summary = chain.invoke(
        input={"information": linkedin_data, "twitter_posts": tweets}
    )

    print(res, linkedin_data.get("photoUrl"))
    return res, linkedin_data.get("photoUrl")


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Harrison Chase", model="ollama", verbose=True)
