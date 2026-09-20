""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langchain.chat_models import init_chat_model

model = init_chat_model(model="claude-haiku-4-5-20251001")

response = model.invoke("What's the capital of the Moon?")

print(response.content)

from pprint import pprint

pprint(response.response_metadata)

# initializing and invoking an agent
from langchain.agents import create_agent

agent = create_agent(model=model)

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)

pprint(response)

print(response['messages'][-1].content)

# adding conversation memory
from langchain.messages import AIMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?"),
    AIMessage(content="The capital of the Moon is Luna City."),
    HumanMessage(content="Interesting, tell me more about Luna City")]}
)

pprint(response)

# streaming the output
for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
        stream_mode="messages"
):

    # token is a message chunk with token content
    # metadata contains which node produced the token

    if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token

