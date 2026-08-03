from google import genai
from google.genai import types
import os
import time
from google.genai._gaos.lib.compat_errors import RateLimitError
from dotenv import find_dotenv, load_dotenv

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize client
client = genai.Client(api_key=gemini_api_key)

for attempt in range(5):
    try:
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input="When is the next F1 race and who is most likely to win it?"
        )
        print(interaction.output_text)
        break
    except RateLimitError:
        wait_time = (2 ** attempt) + 1
        print(f"Quota exceeded. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
