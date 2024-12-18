import os
from openai import OpenAI
client = OpenAI()

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)

print(response["choices"][0]["message"]["content"])