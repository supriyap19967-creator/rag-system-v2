from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    NUMERICAL = "NUMERICAL"
    QUALITATIVE = "QUALITATIVE"
    HYBRID = "HYBRID"


class QueryIntent(BaseModel):
    intent: IntentCategory
    reasoning: str


class SourceCitation(BaseModel):
    filename: str
    page_number: Optional[int] = None


class StructuredAnswer(BaseModel):
    answer: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_citations: List[SourceCitation] = Field(default_factory=list)
