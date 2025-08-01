from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF, DOCX, or email)")
    questions: List[str] = Field(..., description="List of questions to answer about the document")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about insurance policy")
    context: Optional[str] = Field(None, description="Additional context")

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    metadata: Optional[dict] = Field(None, description="Associated metadata")
