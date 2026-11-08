export type ChatStatus = 'completed' | 'failed'

export interface ChatResponse {
  thread_id: string
  answer: string
  status: ChatStatus
  llm_calls: number
  tool_calls: number
  history_count: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface HistoryResponse {
  thread_id: string
  messages: ChatMessage[]
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`HTTP ${response.status}: ${body.slice(0, 240)}`)
  }
  return (await response.json()) as T
}

export async function fetchHistory(threadId: string, signal?: AbortSignal): Promise<ChatMessage[]> {
  const response = await fetch(`/threads/${encodeURIComponent(threadId)}/history`, { signal })
  const data = await readJson<HistoryResponse>(response)
  return data.messages
}

export async function sendChat(threadId: string, message: string): Promise<ChatResponse> {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, message }),
  })
  return readJson<ChatResponse>(response)
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const response = await fetch('/health')
    return response.ok
  } catch {
    return false
  }
}
