import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from linkedin import scrape_linkedin_profile
linkedin_gist = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"

if __name__ == "__main__":
    load_dotenv


    summary_template = """
        given the Linkedin information {information} about a person from I want you to create 
        1. a short summary along with his name, address at top.
        2. two interesting facts about them.
        in Markdown format.
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],template=summary_template)

    llm = ChatOllama(
        base_url="http://localhost:11434",
        model = "llama3.2:3b-instruct-q8_0",  
        temperature = 0,
    )

    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_gist, mock=True)
    res = chain.invoke(input={"information": linkedin_data})
    print(res)