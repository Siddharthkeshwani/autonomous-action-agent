from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, StringConstraints


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Message = Annotated[str, StringConstraints(pattern=r"\S")]


class ChatRequest(BaseModel):
    thread_id: Name
    message: Message
    channel: Literal["chat", "voice"] = "chat"


class ConfirmRequest(BaseModel):
    thread_id: Name
    approved: StrictBool
    edited_args: dict | None = None
    interrupt_id: Name | None = Field(
        default=None, description="Use the ID from hil_required. Required after a previous decision on this thread.",
    )


class Approval(BaseModel):
    thread_id: str
    interrupt_id: str
    action: str
    payload: dict
    tool_name: str
    arguments: dict
    impact_summary: str


class ConfirmResponse(BaseModel):
    thread_id: str
    approved: bool
    status: Literal["completed", "rejected", "awaiting_approval", "error"]
    answer: str | None = None
    receipt: dict | None = None
    approval: Approval | None = None
    error: str | None = None


class VoiceTokenRequest(BaseModel):
    identity: Name
    room_name: Name | None = None


class VoiceTokenResponse(BaseModel):
    url: str
    token: str
    room_name: str
    identity: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, bool]
