from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


RecordId = Annotated[int, Field(gt=0, le=2**63 - 1)]
NonBlank = Annotated[str, StringConstraints(pattern=r"\S")]


class RefundDetails(BaseModel):
    order_id: RecordId
    amount_inr: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    reason: NonBlank
    idempotency_key: Annotated[NonBlank, Field(max_length=128)]


class RefundRequest(RefundDetails):
    approved_by: NonBlank


class TicketDetails(BaseModel):
    customer_id: RecordId
    order_id: RecordId
    category: NonBlank
    priority: Literal["LOW", "NORMAL", "HIGH"]


class TicketRequest(TicketDetails):
    approved_by: NonBlank


class EmailDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    to: NonBlank
    subject: NonBlank
    body: Annotated[str, Field(min_length=1)]


class EmailRequest(EmailDetails):
    approved_by: NonBlank
