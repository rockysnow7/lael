from pydantic import BaseModel, Discriminator, Field
from typing import Annotated, Literal


class Press(BaseModel):
    type: Literal["press"] = "press"
    frequency: float | None = None

class Release(BaseModel):
    type: Literal["release"] = "release"

EventContent = Press | Release

class Event(BaseModel):
    event_content: Annotated[EventContent, Discriminator("type")] = Field(discriminator="type")
    timestamp: float | None = None
