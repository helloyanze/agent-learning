import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { fetchHealth, fetchHistory, sendChat, type ChatMessage, type ChatResponse } from './api'

const STORAGE_KEY = 'task14_agent_thread_id'
const VALID_THREAD_ID = /^[A-Za-z0-9_-]{1,64}$/

function freshThreadId(): string {
  return `web-${crypto.randomUUID().replaceAll('-', '').slice(0, 14)}`
}

function initialThreadId(): string {
  const saved = localStorage.getItem(STORAGE_KEY)
  return saved && VALID_THREAD_ID.test(saved) ? saved : freshThreadId()
}

export default function App() {
  const [threadId, setThreadId] = useState(initialThreadId)
  const [threadDraft, setThreadDraft] = useState(threadId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [healthOk, setHealthOk] = useState<boolean | null>(null)
  const [lastRun, setLastRun] = useState<ChatResponse | null>(null)
  const [pendingQuestion, setPendingQuestion] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const canInteract = !sending && !loadingHistory

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, threadId)
    setThreadDraft(threadId)
    setLastRun(null)
    setError('')
    setMessages([])
    setLoadingHistory(true)
    const controller = new AbortController()
    fetchHistory(threadId, controller.signal)
      .then(setMessages)
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(toMessage(e))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingHistory(false)
      })
    return () => controller.abort()
  }, [threadId])

  useEffect(() => {
    void fetchHealth().then(setHealthOk)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, pendingQuestion, sending])

  function switchThread(next: string) {
    if (!VALID_THREAD_ID.test(next)) {
      setError('会话 ID 仅支持英文字母、数字、_、-，长度为 1–64。')
      return
    }
    setError('')
    setThreadId(next)
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault()
    const question = draft.trim()
    if (!question || !canInteract) return
    if (question.length > 2000) {
      setError('消息最多 2000 个字符。')
      return
    }

    setSending(true)
    setPendingQuestion(question)
    setDraft('')
    setError('')
    try {
      const result = await sendChat(threadId, question)
      setLastRun(result)
      const history = await fetchHistory(threadId)
      setMessages(history)
      if (result.status === 'failed') setError(result.answer || 'Agent 本次执行失败。')
    } catch (e: unknown) {
      // 请求可能到达后端但响应丢失；先尝试同步数据库中实际保留的消息。
      setError(toMessage(e))
      try { setMessages(await fetchHistory(threadId)) } catch { /* 保持原历史 */ }
    } finally {
      setPendingQuestion('')
      setSending(false)
    }
  }

  function onInputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submit()
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">G</div>
          <div><strong>Graph Agent</strong><span>学习项目 · Week 02</span></div>
        </div>

        <div className="sidebar-section">
          <div className="overline">当前会话</div>
          <div className="thread-card">
            <label htmlFor="thread-id">Thread ID</label>
            <input id="thread-id" value={threadDraft} maxLength={64} disabled={!canInteract}
              onChange={(e) => setThreadDraft(e.target.value)} />
            <div className="sidebar-actions">
              <button className="secondary-button" type="button" disabled={!canInteract}
                onClick={() => switchThread(threadDraft.trim())}>切换会话</button>
              <button className="primary-button" type="button" disabled={!canInteract}
                onClick={() => switchThread(freshThreadId())}>+ 新建</button>
            </div>
          </div>
          <p className="aside-hint">输入 <code>api-study-a</code> 可以读取你在 Swagger 中的会话历史。</p>
        </div>

        <div className="sidebar-section">
          <div className="overline">运行状态</div>
          <div className="info-line"><span>FastAPI</span><span className={healthOk === true ? 'online' : 'offline'}>
            <i className="status-dot" />{healthOk === null ? '检查中' : healthOk ? '已连接' : '未连接'}
          </span></div>
          <div className="info-line"><span>已显示消息</span><strong>{messages.length}</strong></div>
          {lastRun && <>
            <div className="info-line"><span>本次 LLM 调用</span><strong>{lastRun.llm_calls}</strong></div>
            <div className="info-line"><span>本次工具调用</span><strong>{lastRun.tool_calls}</strong></div>
          </>}
        </div>

        <div className="sidebar-footer">
          <span className="tiny-dot" /> LangGraph · DeepSeek · SQLite
          <p>仅供本地学习。尚未实现用户登录与访问鉴权。</p>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div><div className="header-title">Agent 对话</div><div className="header-subtitle">会话：{threadId}</div></div>
          <span className="header-pill">API /chat</span>
        </header>

        <section className="messages-area" aria-live="polite">
          {loadingHistory ? (
            <div className="center-state">正在加载会话历史…</div>
          ) : messages.length === 0 && !pendingQuestion ? (
            <div className="empty-state">
              <div className="hero-mark">✦</div>
              <h1>开始与你的 Agent 对话</h1>
              <p>使用已经搭建的 FastAPI + LangGraph 后端。<br />计算请求将通过 DeepSeek 和 Python 工具执行。</p>
              <button className="sample-button" onClick={() => setDraft('请使用 add 计算 18+24')}>试试：计算 18 + 24</button>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message ${message.role}`} key={`${index}-${message.role}`}>
                  <div className="avatar">{message.role === 'user' ? '你' : 'G'}</div>
                  <div className="bubble"><div className="role-label">{message.role === 'user' ? '你' : 'Agent'}</div><div className="message-content">{message.content}</div></div>
                </article>
              ))}
              {pendingQuestion && <>
                <article className="message user pending"><div className="avatar">你</div><div className="bubble"><div className="role-label">你 · 发送中</div><div className="message-content">{pendingQuestion}</div></div></article>
                <article className="message assistant waiting"><div className="avatar">G</div><div className="bubble"><div className="role-label">Agent</div><div className="typing"><b /><b /><b /></div></div></article>
              </>}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-area">
          {error && <div role="alert" className="error-banner">{error}</div>}
          <form className="composer" onSubmit={(e) => void submit(e)}>
            <textarea aria-label="输入消息" value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={onInputKeyDown}
              disabled={!canInteract} rows={2} maxLength={2000}
              placeholder="发送消息给 Agent，例如：把刚才的结果乘以 2…" />
            <button className="send-button" type="submit" disabled={!draft.trim() || !canInteract}>{sending ? '执行中…' : '发送 ↗'}</button>
          </form>
          <div className="composer-tip">Enter 发送 · Shift + Enter 换行 · 消息记录保存在后端 SQLite</div>
        </div>
      </main>
    </div>
  )
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : '请求失败，请检查后端服务是否启动。'
}
