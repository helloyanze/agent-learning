# 09
import pytest
from pydantic import ValidationError

from tool_executor import execute_tool


# 1. 正常参数
def test_add_success() -> None:
    result = execute_tool(
        "add",
        '{"a": 18, "b": 24}',
    )

    assert result == "42"


# 2. 错误参数类型
def test_invalid_arguments() -> None:
    with pytest.raises(ValidationError):
        execute_tool(
            "add",
            '{"a": "hello", "b": 24}',
        )


# 3. 未注册工具
def test_unknown_tool() -> None:
    with pytest.raises(ValueError):
        execute_tool(
            "delete_all_files",
            "{}",
        )


# 4. 不允许多余参数
def test_extra_arguments() -> None:
    with pytest.raises(ValidationError):
        execute_tool(
            "add",
            '{"a": 18, "b": 24, "c": 100}',
        )
