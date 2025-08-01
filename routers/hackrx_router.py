from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import asyncio
from typing import List

from models.request_models import HackRxRequest
from models.response_models import HackRxResponse
from services.multimodal_rag_service import MultimodalRAGService
from config.settings import get_settings

router = APIRouter()
security = HTTPBearer()
settings = get_settings()

# Initialize the multimodal RAG service
rag_service = MultimodalRAGService()

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authentication"""
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@router.post("/run", response_model=HackRxResponse)
async def process_documents_and_questions(
    request: HackRxRequest,
    token: str = Depends(verify_bearer_token)
):
    """
    Main multimodal RAG endpoint for processing documents and answering questions
    
    This endpoint:
    1. Downloads the document from the provided URL
    2. Extracts text, tables, and images using LlamaParse
    3. Chunks the content using LlamaIndex
    4. Generates embeddings using Together AI's BGE-large
    5. Stores in Pinecone vector database
    6. Retrieves relevant context for each question
    7. Generates answers using Llama-3.1-8B-Instruct-Turbo
    8. Returns answers within 30 seconds
    """
    start_time = time.time()
    
    try:
        # Process the document and questions
        answers = await rag_service.process_document_and_questions(
            document_url=request.documents,
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        
        # Check if processing time exceeds 30 seconds
        if processing_time > settings.max_response_time:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Processing time ({processing_time:.2f}s) exceeded maximum allowed time ({settings.max_response_time}s)"
            )
        
        return HackRxResponse(
            answers=answers,
            processing_time=processing_time,
            document_type=rag_service.last_document_type,
            extraction_method="LlamaParse + LlamaIndex + Together AI"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for the HackRx service"""
    try:
        # Check if all services are healthy
        health_status = await rag_service.health_check()
        return {
            "status": "healthy",
            "services": health_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )