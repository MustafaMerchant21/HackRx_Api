import asyncio
import openai
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Example mock input for embed_chunks_parallel
mock_chunks = [
    {
        "chunk_text": "This is the first chunk of text from document A.",
        "metadata": {
            "source": "document_A.pdf",
            "chunk_page_number": 1,
            "chunk_no": 0,
        }
    },
    {
        "chunk_text": "Second chunk, possibly from another page.",
        "metadata": {
            "source": "document_A.pdf",
            "chunk_page_number": 2,
            "chunk_no": 1,
        }
    },
    {
        "chunk_text": "Text from document B, first chunk.",
        "metadata": {
            "source": "document_B.docx",
            "chunk_page_number": 1,
            "chunk_no": 0,
        }
    }
]

class EmbeddingService:
    """Handles embedding generation (Step 3)"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key
        )
        self.model = self.settings.openai_model  # text-embedding-3-small
        self.batch_size = self.settings.embedding_batch_size  # 100
        self.max_concurrent = self.settings.max_concurrent_embeddings  # 5
    
    async def embed_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main function: Generate embeddings for chunks in parallel with optimal batching
        Input format:
        [
            {
                "chunk_text": "Chunked text content",
                "metadata": {
                    "source": "document_name.pdf",
                    "chunk_page_number": 0,
                    "chunk_no": 0,
                }
            }
        ]
        """
        if not chunks:
            return []
        
        logger.info(f"Starting parallel embedding generation for {len(chunks)} chunks")
        
        # Prepare batches for efficient processing
        batches = self.prepare_embedding_batch(chunks, self.batch_size)
        
        logger.debug(f"Created {len(batches)} batches for embedding")
        # # print batches for debugging
        # for idx, batch in enumerate(batches):
        #     logger.debug(f"Batch {idx}: {len(batch)} chunks")
        # return [chunk for batch in batches for chunk in batch]        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Process batches concurrently
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._process_batch_with_semaphore(semaphore, batch, batch_idx)
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions with proper type checking
        embedded_chunks: List[Dict[str, Any]] = []
        
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {batch_idx} failed: {result}")
                # Add empty embeddings for failed batch chunks
                batch = batches[batch_idx]
                for chunk in batch:
                    embedded_chunks.append({
                        **chunk,
                        "embedding": [0.0] * 1536,
                        "embedding_error": str(result)
                    })
            else:
                if isinstance(result, list):
                    embedded_chunks.extend(result)
                else:
                    logger.error(f"Unexpected result type for batch {batch_idx}: {type(result)}")
        
        logger.info(f"Completed embedding generation for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    async def _process_batch_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                          batch: List[Dict], batch_idx: int) -> List[Dict[str, Any]]:
        """Process a single batch with semaphore control"""
        async with semaphore:
            logger.debug(f"Processing batch {batch_idx} with {len(batch)} chunks")
            return await self._process_single_batch(batch)
    
    async def _process_single_batch(self, batch: List[Dict]) -> List[Dict[str, Any]]:
        """Process a single batch of chunks"""
        # Extract texts from batch
        texts = [chunk["chunk_text"] for chunk in batch]
        
        # Get embeddings for the batch
        embeddings = await self.batch_embed_texts(texts)
        
        # Combine embeddings with original chunk data
        embedded_chunks = []
        for chunk, embedding in zip(batch, embeddings):
            embedded_chunks.append({
                **chunk,
                "embedding": embedding,
                "embedding_model": self.model,
                "embedding_dimension": len(embedding) if embedding else 0
            })
        
        return embedded_chunks
    
    async def batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts efficiently using OpenAI API"""
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text.strip()]
            if not valid_texts:
                return [[0.0] * 1536] * len(texts)  # Return default embeddings
            
            # Call OpenAI API with batch of texts
            response = await self.client.embeddings.create(
                model=self.model,
                input=valid_texts,
                encoding_format="float"
            )
            
            # Extract embeddings from response
            embeddings = []
            valid_idx = 0
            
            for original_text in texts:
                if original_text.strip():
                    embeddings.append(response.data[valid_idx].embedding)
                    valid_idx += 1
                else:
                    # Empty text gets default embedding
                    embeddings.append([0.0] * 1536)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return default embeddings on failure
            return [[0.0] * 1536] * len(texts)
    
    async def embed_single_chunk(self, chunk_text: str) -> List[float]:
        """Generate embedding for single text chunk"""
        try:
            if not chunk_text.strip():
                return [0.0] * 1536
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=[chunk_text.strip()],
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Single chunk embedding failed: {e}")
            return [0.0] * 1536
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query text (used in retrieval phase)"""
        try:
            if not query.strip():
                return [0.0] * 1536
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=[query.strip()],
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return [0.0] * 1536
    
    def prepare_embedding_batch(self, chunks: List[Dict], batch_size: int = 100) -> List[List[Dict]]:
        """Prepare chunks for batch processing with optimal batch sizes"""
        if not chunks:
            return []
        
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches from {len(chunks)} chunks")
        return batches

# if __name__ == "__main__":
#     # Example usage
#     embedding_service = EmbeddingService()
    
#     # Simulate parallel embedding generation
#     loop = asyncio.get_event_loop()
#     result = loop.run_until_complete(embedding_service.embed_chunks_parallel(mock_chunks))
    
#     print("Embedded Chunks:")
#     for item in result:
#         print(item)