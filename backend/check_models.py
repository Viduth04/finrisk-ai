from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Models that support generateContent:\n")
for model in gemini.models.list():
    actions = getattr(model, "supported_actions", None)
    if actions is None:
        actions = getattr(model, "supported_generation_methods", [])
    if "generateContent" in actions:
        print(model.name)