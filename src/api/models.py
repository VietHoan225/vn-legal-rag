from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str
    limit: int = 3

class SourceDocument(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
