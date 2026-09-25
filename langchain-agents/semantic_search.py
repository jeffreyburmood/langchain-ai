""" From Langchain Academy - Langchain Agents with Python - Module 2 """

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

# semantic text search using embeddings and vector search
loader = PyPDFLoader("resources/acmecorp-employee-handbook.pdf")

data = loader.load()

print(data)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)

print(len(all_splits))

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many days of vacation does an employee get in their first year?"
)

print(results[0])

# RAG agent
from langchain.tools import tool

@tool
def search_handbook(query: str) -> str:
    """Search the employee handbook for information"""
    results = vector_store.similarity_search(query)
    return results[0].page_content

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[search_handbook],
    system_prompt="You are a helpful agent that can search the employee handbook for information."
    )

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="How many days of vacation does an employee get in their first year?")]}
)

from pprint import pprint

pprint(response)
