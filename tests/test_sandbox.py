import asyncio
import json
from base64 import b64decode
from struct import unpack_from
from time import monotonic
from types import SimpleNamespace
from unittest.mock import MagicMock
from zlib import crc32

import pytest
from e2b import CommandExitException, CommandResult, FileType, Sandbox, TimeoutException
from fastmcp import Client

from services.mcp_server.server import execute_python_sandbox, mcp
from src.sandbox.base import PNG_SIGNATURE


@pytest.fixture
def docker_tool(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    return execute_python_sandbox


def test_sandbox_returns_png(docker_tool):
    code = """\
import json
import matplotlib.pyplot as plt
import pandas as pd

with open("input_data.json") as inputs:
    values = pd.DataFrame(json.load(inputs))["amount"]
print(values.sum())
plt.plot(values)
plt.savefig("/tmp/chart_amount.png")
"""
    result = docker_tool(code, [{"amount": 2}, {"amount": 3}])
    assert result["exit_code"] == 0, result["stderr"]
    assert result["stdout"] == "5\n"
    assert set(result["charts"]) == {"chart_amount.png"}
    png = b64decode(result["charts"]["chart_amount.png"], validate=True)
    assert png.startswith(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    chunks = []
    while offset < len(png):
        size, kind = unpack_from(">I4s", png, offset)
        end = offset + 8 + size
        assert crc32(png[offset + 4:end]) == unpack_from(">I", png, end)[0]
        chunks.append(kind)
        offset = end + 4
    assert chunks[0] == b"IHDR" and chunks[-1] == b"IEND" and b"IDAT" in chunks


def test_sandbox_kills_infinite_loop(docker_tool):
    started = monotonic()
    result = docker_tool('print("started")\nwhile True: pass')
    elapsed = monotonic() - started
    assert 29 <= elapsed < 35
    assert result["stdout"] == "started\n"
    assert result["exit_code"] == 124
    assert "exceeded 30 seconds" in result["stderr"]


def test_sandbox_blocks_network(docker_tool):
    result = docker_tool(
        "import socket\n"
        "assert [name for _, name in socket.if_nameindex()] == ['lo']\n"
        "socket.create_connection(('google.com', 80), timeout=2)\n"
    )
    assert result["exit_code"] == 1
    assert "socket.gaierror:" in result["stderr"]


def test_sandbox_returns_trimmed_key_error(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    code = 'import sys\nfor i in range(35): print(f"debug {i}", file=sys.stderr)\n{}["missing"]'

    async def call_tool():
        async with Client(mcp) as client:
            response = await client.call_tool(
                "execute_python_sandbox", {"code": code, "tool_call_id": "call-1"},
            )
            return response.structured_content

    result = asyncio.run(call_tool())
    assert result["exit_code"] == 1
    assert len(result["stderr"].splitlines()) == 25
    assert "debug 0" not in result["stderr"]
    assert 'File "/workspace/script.py", line 3' in result["stderr"]
    assert result["stderr"].endswith("KeyError: 'missing'")
    assert result["tool_message"] == {
        "type": "tool", "status": "error", "content": result["stderr"],
        "name": "execute_python_sandbox", "tool_call_id": "call-1",
    }


@pytest.fixture
def e2b_sandbox(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "e2b")
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    create = MagicMock()
    monkeypatch.setattr(Sandbox, "create", create)
    sandbox = create.return_value.__enter__.return_value
    sandbox.files.list.return_value = []
    return create, sandbox


def test_e2b_uses_same_tool_interface(e2b_sandbox):
    create, sandbox = e2b_sandbox
    sandbox.commands.run.return_value = CommandResult(stdout="3\n", stderr="", exit_code=0, error=None)
    png = PNG_SIGNATURE + b"test"
    sandbox.files.list.return_value = [SimpleNamespace(
        type=FileType.FILE, symlink_target=None, name="chart_total.png",
        path="/tmp/chart_total.png", size=len(png),
    )]
    sandbox.files.read.return_value = png
    result = execute_python_sandbox("print(3)", {"total": 3})
    assert result["exit_code"] == 0
    assert result["stdout"] == "3\n"
    assert b64decode(result["charts"]["chart_total.png"]) == png
    assert create.call_args.kwargs["allow_internet_access"] is False
    files = sandbox.files.write_files.call_args.args[0]
    assert files[0] == {"path": "/tmp/sandbox/script.py", "data": "print(3)"}
    assert json.loads(files[1]["data"]) == {"total": 3}
    assert sandbox.commands.run.call_args.kwargs["timeout"] == 30
    sandbox.set_timeout.assert_called_once_with(30)
    create.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("error,exit_code,stderr", [
    (TimeoutException("timeout"), 124, "exceeded 30 seconds"),
    (CommandExitException(stderr="debug\n" * 35 + "KeyError: 'x'", stdout="", exit_code=1, error=None),
     1, "KeyError: 'x'"),
])
def test_e2b_returns_failures_and_cleans_up(e2b_sandbox, error, exit_code, stderr):
    create, sandbox = e2b_sandbox
    sandbox.commands.run.side_effect = error
    result = execute_python_sandbox("code", tool_call_id="call-2")
    assert result["exit_code"] == exit_code
    assert stderr in result["stderr"]
    assert len(result["stderr"].splitlines()) <= 25
    create.return_value.__exit__.assert_called_once()
