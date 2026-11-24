from dataclasses import dataclass, field
from typing import Protocol


TIMEOUT_SECONDS = 30
TIMEOUT_MESSAGE = f"Python execution exceeded {TIMEOUT_SECONDS} seconds."
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
InputData = list[dict] | dict | None


class SandboxError(RuntimeError):
    """The sandbox could not run the script or retrieve its output."""


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    charts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.exit_code != 0:
            self.stderr = "\n".join(self.stderr.splitlines()[-25:])


class SandboxRunner(Protocol):
    def run(self, code: str, input_data: InputData = None) -> SandboxResult: ...
