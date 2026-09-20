""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

# tool definition and description
from langchain.tools import tool

# uses tool decorator with descriptive name and doc note
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# can override the function name to make it more descriptive to the agent
@tool("square_root")
def tool1(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# can add a tool description directly to the decorator as well
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5

# can then add the tool in a list to the agent
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[tool1],
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number."
)

from langchain.messages import HumanMessage

question = HumanMessage(content="What is the square root of 467?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

from pprint import pprint

pprint(response['messages'])

print(response["messages"][1].tool_calls)
