import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the GEMINI_API_KEY from your .env file
load_dotenv()
from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Who won the 2026 british f1 gp?",
    tools=[{"type": "google_search"}]
)

print(interaction.output_text)