from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from typing import List
import asyncio

from models.request_models import QueryRequest
from models.response_models import QueryResponse, HealthResponse, ProcessingStatus
from services.document_parser import DocumentParser
from services.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService

router = APIRouter()

# Initialize services
document_parser = DocumentParser()
text_chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()
retrieval_service = RetrievalService()
llm_service = LLMService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # TODO: Add service health checks
    pass

@router.post("/query")
async def process_query_with_documents(
    query: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Main RAG pipeline endpoint - processes documents and returns structured decisions
    """
    try:
        # Step 1: Parse uploaded documents
        parsed_documents = await parse_uploaded_files(files)
        
        # Step 2: Chunk the texts
        chunked_documents = await chunk_documents(parsed_documents)
        
        # Step 3: Generate embeddings (parallel)
        embedded_chunks = await embed_chunks_parallel(chunked_documents)
        
        # Step 4: Store in Pinecone DB
        await store_embeddings_in_pinecone(embedded_chunks)
        
        # Step 5: Search relevant chunks
        relevant_chunks = await search_similar_chunks(query)
        
        # Step 6: Run LLM decision pipeline
        final_decision = await run_llm_decision_pipeline(query, relevant_chunks)
        
        # Step 7: Return structured response
        return QueryResponse(**final_decision)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@router.post("/query-async", response_model=ProcessingStatus)
async def process_query_async(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Async processing endpoint for large documents
    """
    # TODO: Implement async processing with task tracking
    pass

# Pipeline step functions (called by main endpoint)
async def parse_uploaded_files(files: List[UploadFile]):
    """Step 1: Parse and extract text from uploaded files"""
    return await document_parser.parse_files(files)

async def chunk_documents(parsed_documents):
    """Step 2: Chunk documents into smaller pieces"""
    return await text_chunker.chunk_documents(parsed_documents)

async def embed_chunks_parallel(chunked_documents):
    """Step 3: Generate embeddings for chunks in parallel"""
    return await embedding_service.embed_chunks_parallel(chunked_documents)

async def store_embeddings_in_pinecone(embedded_chunks):
    """Step 4: Store embeddings in Pinecone vector database"""
    return await vector_store.store_embeddings(embedded_chunks)

async def search_similar_chunks(query: str):
    """Step 5: Search for semantically similar chunks"""
    return await retrieval_service.search_similar_chunks(query)

async def run_llm_decision_pipeline(query: str, context_chunks):
    """Step 6: Run LLM reasoning pipeline for final decision"""
    return await llm_service.run_decision_pipeline(query, context_chunks)
