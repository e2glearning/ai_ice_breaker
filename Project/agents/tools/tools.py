from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


def get_profile_url_tavily(name: str) -> str:
    search = TavilySearchResults()
    search_result = search.run(verbose=True, tool_input=name)

    return search_result
