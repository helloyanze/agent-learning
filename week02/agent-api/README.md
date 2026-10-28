# Task 14.2：FastAPI × DeepSeek × LangGraph

把 `main.py`、`agent_service.py`、`test_main.py` 放在：

`D:\Desktop\agent-learning\week02\agent-api\`

三个文件都使用完整新版本覆盖，不要追加到原文件末尾。之前的 `week01` 实验文件不需要移动、修改或删除。

## 运行

在 Conda 的 `agent-learning` 环境：

```powershell
cd D:\Desktop\agent-learning\week02\agent-api
python -m pip install "fastapi[standard]" httpx langgraph langgraph-checkpoint-sqlite openai python-dotenv
pytest test_main.py -v
python -m uvicorn main:app --reload
```

DeepSeek API Key 沿用 `week01/llm-basics/.env` 中的 `DEEPSEEK_API_KEY`。

浏览器打开 `http://127.0.0.1:8000/docs`，用 `POST /chat` 依次发送：

```json
{"thread_id":"api-study-a","message":"请使用 add 计算 18+24"}
```

```json
{"thread_id":"api-study-a","message":"把刚才的结果乘以2，请调用 multiply"}
```

预期分别为 42、84。然后通过 `GET /threads/api-study-a/history` 检查简要问答历史。

新建会话 `api-study-b`，发送 `3+4`，预期为 7。

重新启动 Uvicorn 后，请使用同一 thread_id 再次发送依赖前文的问题，测试 SQLite 跨进程恢复。

## 重要说明

- 离线 pytest 使用 FakeService，只验证 HTTP 接口；真正的 DeepSeek 调用由 `/docs` 的实际请求验证。
- 当前使用本地独立数据库 `agent_api_checkpoints.sqlite`，不共享 week01 CLI 的数据库。
- 同步 SQLite Saver + 全局锁仅适合本地教学和低并发演示，不要开启多个 workers。
- API 当前没有登录鉴权，只绑定 `127.0.0.1` 本地开发；不要向公网开放。
- 不要把 `.env` 或数据库文件提交到公开仓库。
