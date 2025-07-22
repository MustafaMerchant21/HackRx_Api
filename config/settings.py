from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_title: str = "HackRx API"
    api_version: str = "1.0.0"
    
    # OpenAI Settings
    openai_api_key: str = "<API-KEY>"
    openai_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"
    
    # Pinecone Settings
    pinecone_api_key: str = "<API-KEY>"
    pinecone_environment: str = "<ENVIRONMENT>"
    pinecone_index_name: str = "insurance-policies"
    
    # Processing Settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 10
    
    # Performance Settings
    embedding_batch_size: int = 100
    max_concurrent_embeddings: int = 5
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
