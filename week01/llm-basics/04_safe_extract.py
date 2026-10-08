import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class CandidateModel(BaseModel):
    name: str
    skills: list[str]
    experience: int


def parse_candidate(content: str) -> CandidateModel | None:
    try:
        # TODO 1
        # JSON 字符串 -> dict

        data = json.loads(content)

        # TODO 2
        # dict -> CandidateModel

        candidate = CandidateModel.model_validate(data)

        # TODO 3
        # 返回 candidate
        return candidate
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{e}")

    except ValidationError as e:
        print(f"候选人数据校验失败：{e}")

    return None


# content = """
# {
#     "name": "Alice",
#     "skills": ["React", "TypeScript"],
#     "experience": 2
# }
# """

# 不合理json,2后面多一个“，”
# JSON 解析失败：Expecting property name enclosed in double quotes: line 6 column 1 (char 84) None
# content = """
# {
#     "name": "Alice",
#     "skills": ["React", "TypeScript"],
#     "experience": 2,
# }
# """

# JSON 合法，但不符合 CandidateModel：
# experience 无法解析为 int
# 候选人数据校验失败：1 validation error for CandidateModel
# experience
#   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='abc', input_type=str]
#     For further information visit https://errors.pydantic.dev/2.13/v/int_parsing
# None
# content = """
# {
#     "name": "Alice",
#     "skills": ["React"],
#     "experience": "abc"
# }
# """
# candidate = parse_candidate(content)

# print(candidate)

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

if content is None:
    print("模型没有返回内容")
else:
    candidate = parse_candidate(content)

    if candidate is not None:
        print(candidate)
