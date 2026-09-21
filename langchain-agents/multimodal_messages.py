""" From Langchain Academy - Langchain Agents with Python - Module 1 """

from dotenv import load_dotenv

load_dotenv()

# text inputs
from langchain.agents import create_agent

agent = create_agent(
    model='gpt-5-nano',
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

from langchain.messages import HumanMessage

question = HumanMessage(content=[
    {"type": "text", "text": "What is the capital of The Moon?"}
])

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

# input an image as part of the message
from PIL import Image
import base64
import io
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the image file
image_path = os.path.join(script_dir, '..', 'resources', 'image.png')

# Open the image
image = Image.open(image_path)

# Convert image to bytes and encode to base64
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_bytes = bytes(buffer.getvalue())
img_b64 = base64.b64encode(image_bytes)

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this capital"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)

