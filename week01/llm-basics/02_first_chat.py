import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "用一句话解释什么是大语言模型。",
        },
    ],
)
print(response)
print("================================")
print(response.choices)
print("================================")
print(response.choices[0])
print("================================")
print(response.choices[0].message.content)
