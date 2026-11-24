import json
import os
from base64 import b64encode
from time import monotonic

from e2b import CommandExitException, FileType, Sandbox, SandboxException, TimeoutException
from httpx import RequestError

from .base import (
    PNG_SIGNATURE, TIMEOUT_MESSAGE, TIMEOUT_SECONDS, InputData, SandboxError, SandboxResult,
)


WORKSPACE = "/tmp/sandbox"
CHART_BYTES = 16 * 1024 * 1024


class E2BRunner:
    def run(self, code: str, input_data: InputData = None) -> SandboxResult:
        """Run the script in a fresh E2B code interpreter sandbox."""
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            raise SandboxError("E2B sandbox requires E2B_API_KEY.")

        try:
            with Sandbox.create(
                "code-interpreter-v1", api_key=api_key, allow_internet_access=False,
                timeout=TIMEOUT_SECONDS, request_timeout=10,
            ) as sandbox:
                sandbox.files.make_dir(WORKSPACE)
                sandbox.files.write_files([
                    {"path": f"{WORKSPACE}/script.py", "data": code},
                    {"path": f"{WORKSPACE}/input_data.json", "data": json.dumps(input_data)},
                ])
                sandbox.set_timeout(TIMEOUT_SECONDS)
                started = monotonic()
                try:
                    result = sandbox.commands.run(
                        f"timeout -s KILL {TIMEOUT_SECONDS}s python -u script.py",
                        cwd=WORKSPACE, timeout=TIMEOUT_SECONDS, request_timeout=10,
                        envs={"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                    )
                except TimeoutException:
                    return SandboxResult("", TIMEOUT_MESSAGE, 124)
                except CommandExitException as error:
                    result = error

                if result.exit_code == 137 and monotonic() - started >= TIMEOUT_SECONDS:
                    stderr = f"{result.stderr.rstrip()}\n{TIMEOUT_MESSAGE}".lstrip("\n")
                    return SandboxResult(result.stdout, stderr, 124)

                charts = {}
                remaining = CHART_BYTES
                for entry in sandbox.files.list("/tmp"):
                    if (
                        entry.type == FileType.FILE and not entry.symlink_target
                        and entry.name.startswith("chart_") and entry.name.endswith(".png")
                        and 0 < entry.size <= remaining
                    ):
                        data = sandbox.files.read(entry.path, format="bytes")
                        if data.startswith(PNG_SIGNATURE) and len(data) <= remaining:
                            charts[entry.name] = b64encode(data).decode("ascii")
                            remaining -= len(data)
                return SandboxResult(result.stdout, result.stderr, result.exit_code, charts)
        except (SandboxException, RequestError):
            raise SandboxError("E2B sandbox could not complete the run.") from None
