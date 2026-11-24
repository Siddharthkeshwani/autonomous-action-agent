import json
from contextlib import closing, suppress
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import monotonic

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import LogConfig, Mount
from requests.exceptions import ConnectionError as DockerConnectionError
from requests.exceptions import ReadTimeout, RequestException
from urllib3.exceptions import ReadTimeoutError

from .base import InputData, SandboxError, SandboxResult, TIMEOUT_MESSAGE, TIMEOUT_SECONDS


IMAGE = "autonomous-action-sandbox:phase4"
# Docker clears /tmp on exit, so write the charts as a final stdout record.
COLLECT_CHARTS = r"""
import base64, json, os, stat
from pathlib import Path

charts = {}
for path in sorted(Path("/tmp").glob("chart_*.png")):
    try:
        with os.fdopen(os.open(path, os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as file:
            if stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                data = file.read()
                if data.startswith(b"\x89PNG\r\n\x1a\n"):
                    charts[path.name] = base64.b64encode(data).decode("ascii")
    except OSError:
        continue
print("\x1e" + json.dumps(charts), flush=True)
"""


class DockerRunner:
    def run(self, code: str, input_data: InputData = None) -> SandboxResult:
        """Run the script in a fresh Linux container."""
        try:
            with TemporaryDirectory() as folder, closing(docker.from_env(timeout=10)) as client:
                mounts = []
                for name, content in (("script.py", code), ("input_data.json", json.dumps(input_data))):
                    path = Path(folder, name)
                    path.write_text(content, encoding="utf-8", newline="\n")
                    path.chmod(0o644)
                    mounts.append(Mount(f"/workspace/{name}", str(path), type="bind", read_only=True))

                command = (
                    f"timeout -s KILL {TIMEOUT_SECONDS}s python -u script.py\n"
                    "status=$?\n"
                    f"python -I -c {quote(COLLECT_CHARTS)}\n"
                    'exit "$status"'
                )
                container = client.containers.create(
                    IMAGE, ["/bin/sh", "-c", command], mounts=mounts,
                    network_disabled=True, network_mode="none", read_only=True,
                    tmpfs={"/tmp": "rw,nosuid,nodev,noexec,size=16m"},
                    mem_limit="512m", memswap_limit="512m", nano_cpus=1_000_000_000,
                    pids_limit=64, ipc_mode="none", cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    log_config=LogConfig(type="json-file", config={"max-size": "32m", "max-file": "1"}),
                )
                try:
                    started = monotonic()
                    container.start()
                    try:
                        remaining = TIMEOUT_SECONDS - (monotonic() - started)
                        if remaining <= 0:
                            raise ReadTimeout()
                        exit_code = container.wait(timeout=remaining)["StatusCode"]
                        timed_out = exit_code == 137 and monotonic() - started >= TIMEOUT_SECONDS
                    except (ReadTimeout, DockerConnectionError) as error:
                        # Requests wraps response body timeouts in ConnectionError.
                        if not isinstance(error, ReadTimeout) and not any(
                            isinstance(arg, ReadTimeoutError) for arg in error.args
                        ):
                            raise
                        try:
                            container.kill()
                        except APIError as kill_error:
                            if kill_error.status_code not in (404, 409):
                                raise
                        timed_out = True

                    stdout = container.logs(stderr=False).decode("utf-8", errors="replace")
                    stderr = container.logs(stdout=False).decode("utf-8", errors="replace")
                    output, separator, charts = stdout.rpartition("\x1e")
                    stdout = output if separator else stdout
                    if timed_out:
                        stderr = f"{stderr.rstrip()}\n{TIMEOUT_MESSAGE}".lstrip("\n")
                        return SandboxResult(stdout, stderr, 124)
                    return SandboxResult(stdout, stderr, exit_code, json.loads(charts) if separator else {})
                finally:
                    with suppress(NotFound):
                        container.remove(force=True)
        except ImageNotFound:
            raise SandboxError(f"Docker sandbox image {IMAGE} has not been built.") from None
        except (DockerException, RequestException, OSError, json.JSONDecodeError) as error:
            raise SandboxError("Docker sandbox could not complete the run.") from error
