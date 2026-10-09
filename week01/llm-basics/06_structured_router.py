import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal


# 定义LLM的输出结构
class RouteDecision(BaseModel):
    route: Literal["frontend", "backend", "fullstack", "unknown"]
    reason: str


# 初始化LLM
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("DEEPSEEK_API_KEY is not set in the environment variables.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

system_prompt = """
你是一名候选人技术方向分类器。

你的任务是根据候选人的工作经历和工作重心，判断候选人更偏向哪个技术方向。

route 只能是以下四个值之一：
-frontend
-backend
-fullstack
如果无法判断，则返回 "unknown"。

判断要求：
1.不要只根据是否出现某个技术关键词进行判断。
2.应该重点考虑候选人的实际工作内容和工作重心。
3.同时出现前端和后端技术，不代表一定是fullstack。
4.只有候选人长期、持续承担前端和后端工作时，才应判断为fullstack。
5.如果信息不足，请返回 "unknown"。

请只返回JSON，不要输出其他解释

JSON格式如下：
{
    "route": "frontend",
    "reason": "判断理由"
}
"""


# 解析LLM输出
def parse_decision(content: str) -> RouteDecision | None:
    try:
        # JSON字符串->py dict
        data = json.loads(content)
        # dict->pydantic model
        decision = RouteDecision.model_validate(data)
        return decision
    except json.JSONDecodeError as e:
        print(f"JSON解析失败：{e}")
    except ValidationError as e:
        print(f"LLM输出校验失败：{e}")

    return None


# LLM调用


def route_candidate(user_text: str) -> RouteDecision | None:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if content is None:
        print("模型没有返回内容")
        return None

    decision = parse_decision(content)
    return decision


# 测试
def main() -> None:

    # 测试 1：主要做前端
    frontend_candidate = """
    我叫 Alice，有两年开发经验。

    平时主要负责公司的 Web 管理后台，
    大部分时间使用 React 和 TypeScript。

    偶尔会使用 Python 写一些简单接口，
    但主要工作还是前端页面开发和交互实现。
    """

    decision = route_candidate(frontend_candidate)

    if decision is not None:
        print("=== 测试 1 ===")
        print(f"方向：{decision.route}")
        print(f"原因：{decision.reason}")

    # 测试 2：主要做后端
    backend_candidate = """
    我叫 Bob，有三年开发经验。

    主要负责后端 API 和业务服务开发，
    平时主要使用 Python、FastAPI 和 PostgreSQL。

    有时候也会使用 Vue 修改一些简单的管理后台页面，
    但主要工作一直是后端开发。
    """

    decision = route_candidate(backend_candidate)

    if decision is not None:
        print("\n=== 测试 2 ===")
        print(f"方向：{decision.route}")
        print(f"原因：{decision.reason}")

    # 测试 3：真正的全栈
    fullstack_candidate = """
    我叫 Charlie，有四年开发经验。

    长期同时负责 Web 前端和后端服务开发。

    前端主要使用 React 和 TypeScript，
    后端主要使用 Python 和 FastAPI。

    从数据库、接口设计到前端页面，
    都是我的长期工作内容。
    """

    decision = route_candidate(fullstack_candidate)

    if decision is not None:
        print("\n=== 测试 3 ===")
        print(f"方向：{decision.route}")
        print(f"原因：{decision.reason}")


if __name__ == "__main__":
    main()
