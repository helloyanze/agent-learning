import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class CandidateModel(BaseModel):
    name: str
    skills: list[str]
    experience: int


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


system_prompt = """
你是一个候选人信息提取器。

请从用户提供的文本中提取候选人信息，并以 JSON 格式输出。

JSON 格式必须是：

{
    "name": "候选人姓名",
    "skills": ["技能1", "技能2"],
    "experience": 工作年限
}

要求：
- name 必须是字符串
- skills 必须是字符串数组
- experience 必须是整数
- 只输出 JSON，不要输出解释
"""

user_text = "我叫Alice，会React和TypeScript，有2年的开发经验。"


response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ],
    response_format={
        "type": "json_object",
    },
)


content = response.choices[0].message.content

# LLM返回内容是str
print(content)
print(type(content))

# json.loads()将str转换为dict
data = json.loads(content)

print(data)
print(type(data))

# 使用Pydantic进行数据验证和模型化
candidate = CandidateModel.model_validate(data)

print(candidate)
print(type(candidate))

# 自然语言
# → LLM
# → 结构化 JSON
# → Python 数据
# → Pydantic 校验
# → 可被程序可靠使用的对象
