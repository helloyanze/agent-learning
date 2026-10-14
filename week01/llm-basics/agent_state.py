# 11

from dataclasses import dataclass, field
from typing import Literal

AgentStatus = Literal["running", "completed", "failed"]


@dataclass
class AgentState:
    # 当前状态
    status: AgentStatus = "running"

    # 已执行的模型调用轮数
    rounds: int = 0

    # 已尝试执行的工具次数
    tool_calls: int = 0

    # 执行限制
    max_rounds: int = 5
    max_tool_calls: int = 3

    # 最近一次错误
    last_error: str | None = None

    # 执行日志  每个对象独立日志
    events: list[str] = field(default_factory=list)


def start_round(state: AgentState) -> bool:
    """尝试开始一轮 LLM 调用。"""

    if state.status != "running":
        return False

    if state.rounds >= state.max_rounds:
        state.status = "failed"
        state.last_error = "达到最大轮数"
        state.events.append(state.last_error)
        return False

    state.rounds += 1
    state.events.append(f"开始第 {state.rounds} 轮")
    return True


def start_tool(state: AgentState, tool_name: str) -> bool:
    """检查工具次数限制，并记录一次工具调用尝试。"""

    if state.status != "running":
        return False

    if state.tool_calls >= state.max_tool_calls:
        state.status = "failed"
        state.last_error = "达到最大工具调用次数"
        state.events.append(state.last_error)
        return False

    state.tool_calls += 1
    state.events.append(f"调用工具：{tool_name}")
    return True


def complete(state: AgentState) -> None:
    """标记 Agent 已完成。"""

    if state.status != "running":
        return

    state.status = "completed"
    state.events.append("任务完成")
