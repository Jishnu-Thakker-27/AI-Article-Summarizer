import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

article=input("Enter the article: ").strip()

load_dotenv()

api_key=os.getenv("API_KEY")

client=genai.Client(api_key=api_key)


config = types.GenerateContentConfig(
    system_instruction = """
    You are an expert article summarizer.
    Use simple English.
    Avoid unnecessary details.
    Be accurate and objective.
    """,
    temperature=0.2,
    max_output_tokens=500
)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=f'''Summarize this article:
    {article}'''
)

print(response.text)

