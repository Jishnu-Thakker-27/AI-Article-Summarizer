import os
import requests
from google import genai
from google.genai import types
import json
from dotenv import load_dotenv

article=input("Enter the article: ").strip()

load_dotenv()

client=genai.Client()

api_key=os.get("API_KEY")

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

