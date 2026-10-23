# 11
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from openai import OpenAIError

import pytest


# 1. 加载 Agent 模块
# 因为文件名以数字开头，使用 importlib 加载
@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-test-key")

    file_path = Path(__file__).parent / "11_error_recovery_agent.py"

    spec = importlib.util.spec_from_file_location(
        "error_recovery_agent",
        file_path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# 2. 构造假的 LLM 响应
def fake_response(
    tool_name=None,
    arguments=None,
    content=None,
    call_id="call_1",
):
    tool_calls = None

    if tool_name is not None:
        tool_calls = [
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=arguments,
                ),
            )
        ]

    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=("tool_calls" if tool_calls else "stop"),
            )
        ]
    )


# 3. 用假的 Client 替代真实 DeepSeek
def mock_client(monkeypatch, agent_module, responses):
    create_mock = Mock(side_effect=responses)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create_mock,
            )
        )
    )

    monkeypatch.setattr(agent_module, "client", fake_client)

    return create_mock


# 测试一：正常完成
def test_agent_success(monkeypatch, agent_module):

    responses = [
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
        ),
        fake_response(
            tool_name="multiply",
            arguments='{"a":42,"b":2}',
            call_id="call_2",
        ),
        fake_response(
            content="计算结果为84",
        ),
    ]

    create_mock = mock_client(
        monkeypatch,
        agent_module,
        responses,
    )

    state = agent_module.AgentState(
        max_rounds=3,
        max_tool_calls=2,
    )

    result = agent_module.run_agent(
        "(18+24)*2",
        state,
    )

    assert result == "计算结果为84"
    assert state.status == "completed"
    assert state.rounds == 3
    assert state.tool_calls == 2
    assert create_mock.call_count == 3


# 测试二：参数错误后恢复
def test_agent_error_recovery(monkeypatch, agent_module):

    responses = [
        # 第一次返回错误参数
        fake_response(
            tool_name="add",
            arguments='{"a":"abc","b":24}',
            call_id="call_bad",
        ),
        # 第二次返回正确参数
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
            call_id="call_fixed",
        ),
        # 最后返回结果
        fake_response(
            content="18加24等于42",
        ),
    ]

    create_mock = mock_client(
        monkeypatch,
        agent_module,
        responses,
    )

    state = agent_module.AgentState(
        max_rounds=3,
        max_tool_calls=2,
    )

    result = agent_module.run_agent(
        "计算18+24",
        state,
    )

    assert result == "18加24等于42"
    assert state.status == "completed"
    assert state.tool_calls == 2
    assert state.last_error is not None

    # 验证错误被记录
    assert any("参数校验失败" in event for event in state.events)

    # 验证错误确实反馈到了模型消息历史
    messages = create_mock.call_args.kwargs["messages"]

    tool_messages = [
        msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"
    ]

    assert len(tool_messages) == 2
    assert "工具调用失败" in tool_messages[0]["content"]
    assert tool_messages[1]["content"] == "42"


def test_max_rounds(monkeypatch, agent_module):
    # 模型要求调用工具，但预算只有一轮
    responses = [
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
        ),
    ]

    create_mock = mock_client(
        monkeypatch,
        agent_module,
        responses,
    )

    state = agent_module.AgentState(
        max_rounds=1,
        max_tool_calls=5,
    )

    result = agent_module.run_agent(
        "计算18+24",
        state,
    )

    assert result is None
    assert state.status == "failed"
    assert state.rounds == 1
    assert state.tool_calls == 1
    assert state.last_error == "达到最大轮数"

    # 确认只调用了一次 LLM
    assert create_mock.call_count == 1


def test_max_tool_calls(monkeypatch, agent_module):
    responses = [
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
        ),
        fake_response(
            tool_name="multiply",
            arguments='{"a":42,"b":2}',
            call_id="call_2",
        ),
    ]

    create_mock = mock_client(
        monkeypatch,
        agent_module,
        responses,
    )

    state = agent_module.AgentState(
        max_rounds=5,
        max_tool_calls=1,
    )

    result = agent_module.run_agent(
        "(18+24)*2",
        state,
    )

    assert result is None
    assert state.status == "failed"
    assert state.tool_calls == 1
    assert state.last_error == "达到最大工具调用次数"
    assert create_mock.call_count == 2


def test_api_failure(monkeypatch, agent_module):
    # 模拟 DeepSeek API 抛出异常
    responses = [
        OpenAIError("模拟 API 调用失败"),
    ]

    create_mock = mock_client(
        monkeypatch,
        agent_module,
        responses,
    )

    state = agent_module.AgentState(
        max_rounds=3,
        max_tool_calls=2,
    )

    result = agent_module.run_agent(
        "计算18+24",
        state,
    )

    assert result is None
    assert state.status == "failed"
    assert state.rounds == 1
    assert state.tool_calls == 0
    assert "LLM API 调用失败" in state.last_error
    assert create_mock.call_count == 1
