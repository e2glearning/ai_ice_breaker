
import requests
from dotenv  import load_dotenv

load_dotenv()

linkedin_gist = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool):
    """
    Scrape info from linkedin profile and use llm to make sense of it.
    """
    if mock:
        response = requests.get(url=linkedin_gist, timeout=10)
        data = response.json().get("person")
        data = {
            k: v 
            for k, v in data.items()
            if v not in ([], "", "", None) 
            and k not in ["certifications"]
        }
    return data


if __name__ == "__main__":
    print(scrape_linkedin_profile(linkedin_profile_url=linkedin_gist, mock=True))