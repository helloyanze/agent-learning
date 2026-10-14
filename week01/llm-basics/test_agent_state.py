# 11

from agent_state import (
    AgentState,
    start_round,
    start_tool,
    complete,
)


def test_initial_state() -> None:
    state = AgentState()

    assert state.status == "running"
    assert state.rounds == 0
    assert state.tool_calls == 0


def test_max_rounds() -> None:
    state = AgentState(max_rounds=1)

    assert start_round(state) is True
    assert start_round(state) is False

    assert state.rounds == 1
    assert state.status == "failed"
    assert state.last_error == "达到最大轮数"


def test_max_tool_calls() -> None:
    state = AgentState(max_tool_calls=1)

    assert start_tool(state, "add") is True
    assert start_tool(state, "multiply") is False

    assert state.tool_calls == 1
    assert state.status == "failed"


def test_complete() -> None:
    state = AgentState()

    start_round(state)
    start_tool(state, "add")
    complete(state)

    assert state.status == "completed"

    # 完成后不允许继续执行
    assert start_round(state) is False
    assert start_tool(state, "add") is False
