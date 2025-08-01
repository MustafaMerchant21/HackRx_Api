from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_title: str = "HackRx API"
    api_version: str = "1.0.0"
    
    # Together AI Settings
    together_api_key: str = "<TOGETHER_API_KEY>"
    together_embedding_model: str = "BAAI/bge-large-en-v1.5"
    together_llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    together_vision_model: str = "meta-llama/Llama-Vision-Free"
    
    # Pinecone Settings
    pinecone_api_key: str = "<PINECONE_API_KEY>"
    pinecone_environment: str = "<PINECONE_ENVIRONMENT>"
    pinecone_index_name: str = "hackrx-multimodal"
    
    # Authentication Settings
    bearer_token: str = "<BEARER_TOKEN>"
    
    # Processing Settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB for multimodal documents
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 15
    max_response_time: int = 30  # seconds
    
    # Performance Settings
    embedding_batch_size: int = 50
    max_concurrent_embeddings: int = 10
    max_concurrent_llm_calls: int = 5
    
    # Multimodal Settings
    enable_image_extraction: bool = True
    enable_table_extraction: bool = True
    enable_vision_analysis: bool = True
    image_quality_threshold: float = 0.7
    
    # LlamaParse Settings
    llama_parse_api_key: Optional[str] = None
    llama_parse_result_type: str = "markdown"
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
