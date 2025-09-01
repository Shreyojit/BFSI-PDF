from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="One of the allowed questions")


class Source(BaseModel):
    page: int
    type: str
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


class IngestResponse(BaseModel):
    s3_uri: str
    pages: int
    blocks: int
    index_size: int
