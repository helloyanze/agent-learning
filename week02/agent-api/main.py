"""Task 14.2: FastAPI routes backed by the real DeepSeek/LangGraph Agent.

Run locally (from week02/agent-api):
    python -m uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator


DB_PATH = Path(__file__).with_name("agent_api_checkpoints.sqlite")


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空白字符")
        return value


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    status: Literal["completed", "failed"]
    llm_calls: int
    tool_calls: int
    history_count: int


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryItem]


def create_app(service_factory: Callable | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service_factory is not None:
            # 自动化测试使用 FakeService，不访问 DeepSeek 或真实 SQLite。
            app.state.agent_service = service_factory()
            try:
                yield
            finally:
                del app.state.agent_service
        else:
            # 仅在真实启动时加载 Agent 依赖，离线 API 单测无需真实模型。
            from langgraph.checkpoint.sqlite import SqliteSaver
            from agent_service import create_service
            # 开服时创建共享 SQLite 连接；关闭时自动释放。
            with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
                app.state.agent_service = create_service(saver)
                try:
                    yield
                finally:
                    del app.state.agent_service

    app = FastAPI(title="LangGraph Agent API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request):
        # 同步路由运行在线程池中，不阻塞 FastAPI 主事件循环。
        return request.app.state.agent_service.ask(
            thread_id=payload.thread_id,
            message=payload.message,
        )

    @app.get("/threads/{thread_id}/history", response_model=HistoryResponse)
    def thread_history(thread_id: str, request: Request):
        # 本地学习用途：正式多用户部署必须增加身份认证和资源鉴权。
        messages = request.app.state.agent_service.history(thread_id)
        return {"thread_id": thread_id, "messages": messages}

    return app


app = create_app()
