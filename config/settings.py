from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_title: str = "HackRx API"
    api_version: str = "1.0.0"
    
    # Together AI Settings (Primary LLM Provider)
    together_api_key: str = ""
    together_embedding_model: str = "BAAI/bge-large-en-v1.5"
    together_chat_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    together_vision_model: str = "meta-llama/Llama-Vision-Free"
    
    # LlamaParse Settings (Advanced Document Parsing)
    llamaparse_api_key: str = ""
    llamaparse_result_type: str = "markdown"  # or "text"
    llamaparse_verbose: bool = True
    llamaparse_language: str = "en"
    
    # OpenAI Settings (Fallback)
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"
    
    # Pinecone Settings
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "hackrx-multimodal-rag"
    pinecone_dimension: int = 1024  # BGE-large embedding dimension
    pinecone_metric: str = "cosine"
    
    # Authentication
    bearer_token: str = "hackrx-secure-token-2024"
    
    # Processing Settings - Optimized for sub-30-second response
    max_file_size: int = 50 * 1024 * 1024  # 50MB for larger PDFs
    chunk_size: int = 512  # Smaller chunks for better precision
    chunk_overlap: int = 50  # Reduced overlap for faster processing
    max_chunks_per_query: int = 8  # Optimized retrieval count
    
    # Performance Settings - Aggressive parallelization
    embedding_batch_size: int = 32  # Reduced for faster processing
    max_concurrent_embeddings: int = 10  # Increased concurrency
    max_concurrent_llm_calls: int = 5
    
    # Response Time Optimization
    retrieval_timeout: float = 5.0  # Max 5 seconds for retrieval
    llm_timeout: float = 15.0  # Max 15 seconds for LLM processing
    total_pipeline_timeout: float = 28.0  # Total pipeline timeout
    
    # Caching Settings
    enable_embedding_cache: bool = True
    cache_ttl: int = 3600  # 1 hour cache
    
    # Table Extraction Settings
    table_extraction_strategy: str = "hybrid"  # "pdfplumber", "camelot", "hybrid"
    enable_ocr: bool = True
    ocr_confidence_threshold: float = 0.7
    
    # Image Processing Settings
    image_preprocessing: bool = True
    max_image_size: tuple = (1024, 1024)
    image_quality: int = 85
    
    # Multimodal Settings
    enable_vision_processing: bool = True
    vision_processing_timeout: float = 10.0
    
    # Document Processing Timeouts
    pdf_processing_timeout: float = 8.0
    docx_processing_timeout: float = 3.0
    table_extraction_timeout: float = 5.0
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
