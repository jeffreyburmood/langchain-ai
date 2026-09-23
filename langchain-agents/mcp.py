""" From Langchain Academy - Langchain Agents with Python - Module 2 """

from dotenv import load_dotenv

load_dotenv()

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint


async def main():
    client = MultiServerMCPClient(
        {
            "local_server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["../resources/2.1_mcp_server.py"],
                }
        }
    )

    # get tools
    tools = await client.get_tools()

    # get resources
    resources = await client.get_resources("local_server")

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    agent = create_agent(
        model="claude-haiku-4-5-20251001",
        tools=tools,
        system_prompt=prompt
    )

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )

    pprint(response)

    # running a local MCP server

    client = MultiServerMCPClient(
        {
            "local_server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["resources/2.1_mcp_server.py"],
                }
        }
    )

    # get tools
    tools = await client.get_tools()

    # get resources
    resources = await client.get_resources("local_server")

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt=prompt
    )

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main())