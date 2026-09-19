from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfirmedTicketUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registration_date: str | None = Field(default=None, max_length=100)
    user_name: str | None = Field(default=None, max_length=300)
    service: str | None = Field(default=None, max_length=300)
    component: str | None = Field(default=None, max_length=300)
    category: str | None = Field(default=None, max_length=500)
    request_type: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    criticality: str | None = Field(default=None, max_length=100)
    urgency: str | None = Field(default=None, max_length=100)
    priority: str | None = Field(default=None, max_length=100)
    service_class: str | None = Field(default=None, max_length=200)
    timezone: str | None = Field(default=None, max_length=100)
    final_line: str | None = Field(default=None, max_length=500)
    result: str | None = Field(default=None, max_length=10000)
    overdue: bool | None = None
    actual_duration_seconds: int | None = Field(default=None, ge=0, le=31_536_000)
    clarifications_count: int | None = Field(default=None, ge=0, le=100_000)

    @field_validator(
        "registration_date", "service", "category", "request_type", "description",
        "criticality", "urgency", "priority", "service_class", "timezone", "final_line", "result",
    )
    @classmethod
    def strip_required_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Поле не может быть пустым")
        return value

    @field_validator("user_name", "component")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None
