from pydantic import BaseModel, Field
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about insurance policy")
    context: Optional[str] = Field(None, description="Additional context")

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    metadata: Optional[dict] = Field(None, description="Associated metadata")
