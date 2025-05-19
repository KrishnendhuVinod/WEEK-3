import os
import validators
from dotenv import load_dotenv
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


#Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

genai.configure(api_key=GEMINI_API_KEY)

#Web Browser using Selenium
def fetch_web_content(url):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    driver.get(url)
    content = driver.page_source
    driver.quit()

    #Use BeautifulSoup to extract text 
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text

#Summarizer
def summarize_text(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Summarize this:\n{text[:8000]}"
    response = model.generate_content(prompt)
    return response.text

#Researcher
class Researcher:
    def run(self, url):
        print("\n[Researcher] Fetching content from:", url)
        return fetch_web_content(url)

#Summarizer 
class Summarizer:
    def run(self, content):
        print("\n[Summarizer] Generating summary...")
        return summarize_text(content)

#Group Chat Coordinator
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents

    def run(self, input_data):
        data = input_data
        for agent in self.agents:
            data = agent.run(data)
        return data

#Main function
def main():
    url = input("Enter a URL to research and summarize: ").strip()
    if not validators.url(url):
       print("Invalid URL. Please enter a valid URL.")
       return
    
    researcher = Researcher()
    summarizer = Summarizer()
    
    group_chat = RoundRobinGroupChat([researcher, summarizer])
    
    summary = group_chat.run(url)
    
    print("\n=== Final Summary ===\n")
    print(summary)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
        print("\nSummary saved to 'summary.txt'")

if __name__ == "__main__":
    main()
