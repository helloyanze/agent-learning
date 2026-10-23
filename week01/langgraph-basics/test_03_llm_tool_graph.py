import importlib.util
from openai import OpenAIError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def agent_module(monkeypatch):
    # 确保测试不需要真实 API Key
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    path = Path(__file__).parent / "03_llm_tool_graph.py"

    spec = importlib.util.spec_from_file_location(
        "llm_tool_graph",
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ]
    )


def test_graph_success(monkeypatch, agent_module):

    # 模拟三轮 LLM 响应
    responses = [
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
            call_id="call_1",
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

    create_mock = Mock(side_effect=responses)

    monkeypatch.setattr(
        agent_module.client.chat.completions,
        "create",
        create_mock,
    )

    initial_state = {
        "messages": [{"role": "user", "content": "计算(18+24)*2"}],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    result = agent_module.graph.invoke(initial_state)

    assert result["status"] == "completed"
    assert result["llm_calls"] == 3
    assert result["tool_calls"] == 2
    assert result["final_answer"] == "计算结果为84"
    assert create_mock.call_count == 3

    # 验证最后一次请求确实包含工具执行结果
    last_messages = create_mock.call_args.kwargs["messages"]

    tool_messages = [
        msg
        for msg in last_messages
        if isinstance(msg, dict) and msg.get("role") == "tool"
    ]

    assert [msg["content"] for msg in tool_messages] == [
        "42",
        "84",
    ]


def test_graph_error_recovery(monkeypatch, agent_module):

    responses = [
        # 第一次：故意提供错误参数
        fake_response(
            tool_name="add",
            arguments='{"a":"abc","b":24}',
            call_id="call_bad",
        ),
        # 第二次：模型修正参数
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
            call_id="call_fixed",
        ),
        # 第三次：返回最终回答
        fake_response(
            content="计算结果为42",
        ),
    ]

    create_mock = Mock(side_effect=responses)

    monkeypatch.setattr(
        agent_module.client.chat.completions,
        "create",
        create_mock,
    )

    initial_state = {
        "messages": [{"role": "user", "content": "计算18+24"}],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    result = agent_module.graph.invoke(initial_state)

    assert result["status"] == "completed"
    assert result["llm_calls"] == 3
    assert result["tool_calls"] == 2
    assert result["final_answer"] == "计算结果为42"

    # 检查模型最后一次收到的消息
    messages = create_mock.call_args.kwargs["messages"]

    tool_messages = [
        msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"
    ]

    assert len(tool_messages) == 2
    assert "工具调用失败" in tool_messages[0]["content"]
    assert tool_messages[1]["content"] == "42"


def test_graph_max_tool_calls(monkeypatch, agent_module):

    monkeypatch.setattr(
        agent_module,
        "MAX_TOOL_CALLS",
        1,
    )

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

    create_mock = Mock(side_effect=responses)

    monkeypatch.setattr(
        agent_module.client.chat.completions,
        "create",
        create_mock,
    )

    initial_state = {
        "messages": [{"role": "user", "content": "计算(18+24)*2"}],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    result = agent_module.graph.invoke(initial_state)

    assert result["status"] == "failed"
    assert result["llm_calls"] == 2
    assert result["tool_calls"] == 1
    assert result["final_answer"] == "达到最大工具调用次数"
    assert create_mock.call_count == 2


def test_graph_max_llm_calls(monkeypatch, agent_module):

    monkeypatch.setattr(
        agent_module,
        "MAX_LLM_CALLS",
        1,
    )

    responses = [
        fake_response(
            tool_name="add",
            arguments='{"a":18,"b":24}',
        ),
    ]

    create_mock = Mock(side_effect=responses)

    monkeypatch.setattr(
        agent_module.client.chat.completions,
        "create",
        create_mock,
    )

    initial_state = {
        "messages": [{"role": "user", "content": "计算18+24"}],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    result = agent_module.graph.invoke(initial_state)

    assert result["status"] == "failed"
    assert result["llm_calls"] == 1
    assert result["tool_calls"] == 1
    assert result["final_answer"] == "达到最大 LLM 调用次数"
    assert create_mock.call_count == 1


def test_graph_api_failure(monkeypatch, agent_module):

    # 模拟 API 报错，不发送真实请求
    create_mock = Mock(side_effect=OpenAIError("模拟 API 调用失败"))

    monkeypatch.setattr(
        agent_module.client.chat.completions,
        "create",
        create_mock,
    )

    initial_state = {
        "messages": [{"role": "user", "content": "计算18+24"}],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    result = agent_module.graph.invoke(initial_state)

    assert result["status"] == "failed"
    assert result["llm_calls"] == 1
    assert result["tool_calls"] == 0

    assert "LLM API 调用失败" in result["final_answer"]
    assert create_mock.call_count == 1
