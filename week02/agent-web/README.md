# Task 14.3 — React + FastAPI + LangGraph 联调

这是一个独立的 React + TypeScript + Vite 前端示例。后端沿用 `week02/agent-api` 的 Task 14.2 代码包，**无需改动后端文件**。

## 准备环境

- 检查 `node -v`，本项目 Vite 7 需要 Node.js 20.19+ 或 22.12+。如旧项目依赖 Node 16，请使用版本管理器在这个项目切换新版本，不要强制升级旧项目。
- 运行后端（终端 A）：

```powershell
cd D:\Desktop\agent-learning\week02\agent-api
python -m uvicorn main:app --reload
```

- 将本包 `agent-web` 整个文件夹放入 `week02`，与 `agent-api` 同级。启动前端（终端 B）：

```powershell
cd D:\Desktop\agent-learning\week02\agent-web
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。

## 验证

1. 保持默认新会话，输入“请使用 add 计算 18+24”，应得到 42。
2. 再输入“把刚才结果乘以 2，请使用 multiply 工具”，应得到 84。
3. 在侧栏将会话 ID 修改为 `api-study-a` 并点击“切换会话”，应读取原先 Swagger 测试产生的两轮历史。
4. 点击“+ 新建”，应显示空历史，新的 thread_id 独立。
5. 可以运行 `npm run build` 验证 TypeScript 和生产构建。

前端通过 `vite.config.ts` 内的 dev proxy 代理 `/chat`、`/threads` 和 `/health` 到 `http://127.0.0.1:8000`，避免开发环境跨域问题。这个代理**只用于 Vite 本地开发或预览环境**；部署时需由反向代理配置相同路径或在后端显式处理 CORS。

## 结构与学习要点

- `src/api.ts`：统一 `fetch` 封装和返回类型
- `src/App.tsx`：`useState`、`useEffect`、事件处理、请求状态、会话切换
- `vite.config.ts`：开发服务器代理
- `src/index.css`：响应式 UI

目前仍是本地实验项目：没有身份认证、并发请求协调、消息流式输出（SSE）或会话权限控制。不要把 API 暴露到公网；持久化数据由 FastAPI 端的 SQLite 保存，而浏览器仅保存最近选定的会话 ID。
