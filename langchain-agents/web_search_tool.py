""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent

# without use of a web search tool
agent = create_agent(
    model="claude-haiku-4-5-20251001"
)

from langchain.messages import HumanMessage

question = HumanMessage(content="How up to date is your training knowledge?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

# now add a web search tool like Tavily
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

# web_search.invoke("Who is the current mayor of San Francisco?")

agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=[web_search]
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {"messages": [question]}
)

from pprint import pprint

pprint(response['messages'])
