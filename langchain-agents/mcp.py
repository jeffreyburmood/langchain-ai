""" From Langchain Academy - Langchain Agents with Python - Module 2 """

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

import sys
import asyncio

# from langchain_mcp_adapters.client import MultiServerMCPClient

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

    from langchain.agents import create_agent

    agent = create_agent(
        model="claude-haiku-4-5-20251001",
        tools=tools,
        system_prompt=prompt
    )

    from langchain.messages import HumanMessage

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )

    from pprint import pprint

    pprint(response)

if __name__ == "__main__":
    asyncio.run(main())