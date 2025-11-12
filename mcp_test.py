import os

import anthropic

# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()  # loads from .env in the cwd

api_key = os.getenv("ANTHROPIC_API_KEY")

# Their API key
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Your server URL
MCP_SERVER_URL = "https://27f4a7c3648d.ngrok.app/sse"

# Use Claude with your MCP server
response = client.beta.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "hi, can you list all latent features and retrieve the top activating examples for the first one and tell me?",
        }
    ],
    mcp_servers=[{"type": "url", "url": MCP_SERVER_URL, "name": "delphi"}],
    betas=["mcp-client-2025-04-04"],
)

print(response.content[0].text)
