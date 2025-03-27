from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


def get_profile_url_tavily(username: str) -> str:
    search = TavilySearchResults(
        max_results=5,
        include_answer=True,
    )
    search_result = search.run(verbose=True, tool_input=username)

    return search_result
