from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IncomingManualRequest(BaseModel):
    request_id: str | None = None
    registration_date: str | None = None
    user: str = ""
    service: str = ""
    component: str = ""
    request_type: str = ""
    description: str = Field(default="", max_length=5000)
    criticality: str = ""
    urgency: str = ""
    priority: str = ""
    service_class: str = ""
    timezone: str = ""

    @model_validator(mode="after")
    def require_analyzable_input(self) -> "IncomingManualRequest":
        if not any(
            str(getattr(self, field)).strip()
            for field in (
                "description", "service", "component", "request_type", "criticality", "urgency",
                "priority", "service_class", "timezone",
            )
        ):
            raise ValueError("Укажите текст обращения или регистрационные признаки")
        return self


class CategoryDecisionRequest(BaseModel):
    category: str = Field(min_length=1, max_length=500)


class RouteDecisionRequest(BaseModel):
    route: str = Field(min_length=1, max_length=500)


class IncomingTicketUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    registration_date: str | None = None
    user: str | None = None
    service: str | None = None
    component: str | None = None
    request_type: str | None = None
    description: str | None = Field(default=None, max_length=5000)
    criticality: str | None = None
    urgency: str | None = None
    priority: str | None = None
    service_class: str | None = None
    timezone: str | None = None
