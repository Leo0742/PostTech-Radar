from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TicketAnalyzeRequest(BaseModel):
    description: str = Field(default="", max_length=5000)
    registration_date: str = Field(default="", max_length=100)
    user: str = Field(default="", max_length=300)
    service: str = Field(default="", max_length=300)
    component: str = Field(default="", max_length=300)
    request_type: str = Field(default="", max_length=200)
    criticality: str = Field(default="", max_length=100)
    urgency: str = Field(default="", max_length=100)
    priority: str = Field(default="", max_length=100)
    service_class: str = Field(default="", max_length=200)
    timezone: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def require_analyzable_input(self) -> "TicketAnalyzeRequest":
        if not any(
            str(getattr(self, field)).strip()
            for field in (
                "description", "service", "component", "request_type", "criticality", "urgency",
                "priority", "service_class", "timezone",
            )
        ):
            raise ValueError("Укажите текст обращения или регистрационные признаки")
        return self
