""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

# no memory for chat discussion
from langchain.agents import create_agent


agent = create_agent(
    "claude-haiku-4-5-20251001"
)

from langchain.messages import HumanMessage

question = HumanMessage(content="Hello my name is Jeffrey and my favourite colour is green")

response = agent.invoke(
    {"messages": [question]}
)

from pprint import pprint

pprint(response)

question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]}
)

pprint(response)

