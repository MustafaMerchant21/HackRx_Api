import asyncio
import aiohttp
import tempfile
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from llama_index.multi_modal_llms.together import TogetherMultiModalLLM
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from services.multimodal_processor import MultimodalProcessor

import pinecone
from config.settings import get_settings

settings = get_settings()

class MultimodalRAGService:
    """Main service for multimodal RAG processing using LlamaIndex, LlamaParse, and Together AI"""
    
    def __init__(self):
        self.settings = get_settings()
        self.last_document_type = None
        
        # Initialize Together AI components
        self.embedding_model = TogetherEmbedding(
            model_name=self.settings.together_embedding_model,
            api_key=self.settings.together_api_key
        )
        
        self.llm = TogetherLLM(
            model=self.settings.together_llm_model,
            api_key=self.settings.together_api_key,
            temperature=0.1,
            max_tokens=2048
        )
        
        self.vision_llm = TogetherMultiModalLLM(
            model=self.settings.together_vision_model,
            api_key=self.settings.together_api_key,
            temperature=0.1,
            max_tokens=2048
        )
        
        # Initialize multimodal processor
        self.multimodal_processor = MultimodalProcessor()
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.settings.pinecone_api_key,
            environment=self.settings.pinecone_environment
        )
        
        # Set up LlamaIndex settings
        Settings.embed_model = self.embedding_model
        Settings.llm = self.llm
        Settings.chunk_size = self.settings.chunk_size
        Settings.chunk_overlap = self.settings.chunk_overlap
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            pinecone_index=pinecone.Index(self.settings.pinecone_index_name),
            dimension=1024  # BGE-large dimension
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
    
    async def download_document(self, url: str) -> str:
        """Download document from URL to temporary file"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download document: {response.status}")
                
                # Create temporary file
                suffix = self._get_file_extension(url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    content = await response.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                return temp_file_path
    
    def _get_file_extension(self, url: str) -> str:
        """Extract file extension from URL"""
        if url.endswith('.pdf'):
            return '.pdf'
        elif url.endswith('.docx'):
            return '.docx'
        elif url.endswith('.doc'):
            return '.doc'
        elif url.endswith('.eml'):
            return '.eml'
        else:
            return '.pdf'  # Default to PDF
    
    async def parse_document(self, file_path: str) -> List[Document]:
        """Parse document using advanced multimodal processing"""
        try:
            # Use advanced multimodal processor
            documents = await self.multimodal_processor.process_document_advanced(file_path)
            
            # Optimize chunking
            chunked_documents = await self.multimodal_processor.chunk_documents_optimized(documents)
            
            self.last_document_type = self._detect_document_type(file_path)
            return chunked_documents
            
        except Exception as e:
            raise Exception(f"Failed to parse document: {str(e)}")
    
    def _detect_document_type(self, file_path: str) -> str:
        """Detect document type from file path"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.docx':
            return 'docx'
        elif ext == '.doc':
            return 'doc'
        elif ext == '.eml':
            return 'email'
        else:
            return 'unknown'
    
    async def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create vector index from documents"""
        try:
            # Parse documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Create index
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            return index
            
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
    
    async def answer_questions(self, index: VectorStoreIndex, questions: List[str]) -> List[str]:
        """Answer questions using the created index"""
        try:
            answers = []
            
            # Process questions in parallel for better performance
            tasks = []
            for question in questions:
                task = self._answer_single_question(index, question)
                tasks.append(task)
            
            # Wait for all answers with timeout
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.settings.max_response_time - 5  # Leave 5 seconds buffer
            )
            
            # Handle any exceptions
            processed_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    processed_answers.append(f"Error processing question {i+1}: {str(answer)}")
                else:
                    processed_answers.append(answer)
            
            return processed_answers
            
        except asyncio.TimeoutError:
            raise Exception("Question answering timed out")
        except Exception as e:
            raise Exception(f"Failed to answer questions: {str(e)}")
    
    async def _answer_single_question(self, index: VectorStoreIndex, question: str) -> str:
        """Answer a single question using the index"""
        try:
            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=self.settings.max_chunks_per_query,
                response_mode="compact"
            )
            
            # Get response
            response = await query_engine.aquery(question)
            return response.response
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    async def process_document_and_questions(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> List[str]:
        """Main pipeline: download, parse, index, and answer questions"""
        temp_file_path = None
        
        try:
            # Step 1: Download document
            temp_file_path = await self.download_document(document_url)
            
            # Step 2: Parse document
            documents = await self.parse_document(temp_file_path)
            
            # Step 3: Create index
            index = await self.create_index(documents)
            
            # Step 4: Answer questions
            answers = await self.answer_questions(index, questions)
            
            return answers
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all services"""
        health_status = {}
        
        try:
            # Check Together AI
            test_embedding = await self.embedding_model.aget_text_embedding("test")
            health_status["together_ai"] = "healthy"
        except Exception as e:
            health_status["together_ai"] = f"unhealthy: {str(e)}"
        
        try:
            # Check Pinecone
            pinecone.describe_index(self.settings.pinecone_index_name)
            health_status["pinecone"] = "healthy"
        except Exception as e:
            health_status["pinecone"] = f"unhealthy: {str(e)}"
        
        try:
            # Check LlamaParse
            if self.settings.llama_parse_api_key:
                health_status["llama_parse"] = "healthy"
            else:
                health_status["llama_parse"] = "no_api_key"
        except Exception as e:
            health_status["llama_parse"] = f"unhealthy: {str(e)}"
        
        return health_status