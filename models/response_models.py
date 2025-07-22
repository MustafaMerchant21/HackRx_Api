from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ClauseUsed(BaseModel):
    clause_id: str = Field(..., description="Unique clause identifier")
    document: str = Field(..., description="Source document")
    text: str = Field(..., description="Clause text")
    confidence: float = Field(..., ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    decision: str = Field(..., description="approved/rejected/pending")
    amount: Optional[int] = Field(None, description="Approved amount")
    justification: str = Field(..., description="Decision explanation")
    clauses_used: List[ClauseUsed] = Field(default_factory=list)
    processing_time: Optional[float] = Field(None)
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)

class ProcessingStatus(BaseModel):
    task_id: str = Field(..., description="Processing task ID")
    status: str = Field(..., description="processing/completed/failed")
    message: Optional[str] = Field(None)
