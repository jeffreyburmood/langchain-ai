""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage

# no prompt so the agent answers factually
agent = create_agent(model="claude-haiku-4-5-20251001")

question = HumanMessage(content="What's the capital of the moon?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][1].content)

# now add a prompt to provide soe context for the answer
system_prompt = "You are a science fiction writer, create a capital city at the users request."

scifi_agent = create_agent(
    model="claude-haiku-4-5-20251001",
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response['messages'][1].content)

# few shot examples
system_prompt = """

You are a science fiction writer, create a space capital city at the users request.

User: What is the capital of mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia

"""

scifi_agent = create_agent(
    model="claude-haiku-4-5-20251001",
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response['messages'][1].content)

# prompts to create structured output
system_prompt = """

You are a science fiction writer, create a space capital city at the users request.

Please keep to the below structure.

Name: The name of the capital city

Location: Where it is based

Vibe: 2-3 words to describe its vibe

Economy: Main industries

"""

scifi_agent = create_agent(
    model="claude-haiku-4-5-20251001",
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response['messages'][1].content)

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model="claude-haiku-4-5-20251001",
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
    response_format=CapitalInfo
)

question = HumanMessage(content="What is the capital of The Moon?")

response = agent.invoke(
    {"messages": [question]}
)

print(response["structured_response"])

print(response["structured_response"].name)

capital_info = response["structured_response"]

capital_name = capital_info.name
capital_location = capital_info.location

print(f"{capital_name} is a city located at {capital_location}")