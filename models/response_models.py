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

class HackRxResponse(BaseModel):
    """Response model for /hackrx/run endpoint"""
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the input questions"
    )
    success: bool = Field(True, description="Processing success status")
    processing_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional processing information (chunks processed, models used, etc.)"
    )
    processing_time: Optional[float] = Field(
        None, 
        description="Total processing time in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date.",
                    "There is a waiting period of thirty-six (36) months for pre-existing diseases.",
                    "Yes, the policy covers maternity expenses with 24 months continuous coverage requirement."
                ],
                "success": True,
                "processing_info": {
                    "documents_processed": 1,
                    "chunks_generated": 156,
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "chat_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                },
                "processing_time": 24.5
            }
        }
