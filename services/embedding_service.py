import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from together import Together
import time
from cachetools import TTLCache
import hashlib

from config.settings import get_settings

settings = get_settings()

class EmbeddingService:
    """Enhanced embedding service using Together AI BGE-large with performance optimization"""
    
    def __init__(self):
        self.client = Together(api_key=settings.together_api_key)
        self.model = settings.together_embedding_model  # BAAI/bge-large-en-v1.5
        self.batch_size = settings.embedding_batch_size  # 32
        self.max_concurrent = settings.max_concurrent_embeddings  # 10
        
        # Caching for embeddings to improve performance
        if settings.enable_embedding_cache:
            self.embedding_cache = TTLCache(maxsize=1000, ttl=settings.cache_ttl)
        else:
            self.embedding_cache = None
    
    async def embed_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main embedding function - generates embeddings for chunks in parallel
        Input: List of chunk dictionaries with 'chunk_text' and 'metadata'
        Output: List of chunk dictionaries with added 'embedding' field
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk.get("chunk_text", "") for chunk in chunks]
        
        # Filter out empty texts
        valid_chunks = []
        valid_texts = []
        for i, (chunk, text) in enumerate(zip(chunks, texts)):
            if text and text.strip():
                valid_chunks.append(chunk)
                valid_texts.append(text.strip())
        
        if not valid_texts:
            return []
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(valid_texts)
        
        # Add embeddings to chunks
        result_chunks = []
        for chunk, embedding in zip(valid_chunks, embeddings):
            # Create enhanced chunk with embedding
            enhanced_chunk = chunk.copy()
            enhanced_chunk["embedding"] = embedding
            enhanced_chunk["metadata"]["embedding_model"] = self.model
            enhanced_chunk["metadata"]["embedding_dimension"] = len(embedding)
            enhanced_chunk["metadata"]["embedding_timestamp"] = time.time()
            
            result_chunks.append(enhanced_chunk)
        
        return result_chunks
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Together AI BGE-large model with parallel processing"""
        if not texts:
            return []
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.embedding_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    cached_embeddings.append((i, self.embedding_cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Process uncached texts in batches
        new_embeddings = []
        if uncached_texts:
            batches = [uncached_texts[i:i + self.batch_size] for i in range(0, len(uncached_texts), self.batch_size)]
            
            # Parallel batch processing
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._generate_embedding_batch(batch, semaphore) for batch in batches]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    print(f"Embedding batch failed: {batch_result}")
                    # Add zero embeddings as fallback
                    batch_size = len(batches[0]) if batches else 0
                    new_embeddings.extend([[0.0] * 1024] * batch_size)  # BGE-large has 1024 dimensions
                else:
                    new_embeddings.extend(batch_result)
            
            # Cache new embeddings
            if self.embedding_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.embedding_cache[cache_key] = embedding
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for uncached_idx, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[uncached_idx] = embedding
        
        return all_embeddings
    
    async def _generate_embedding_batch(self, texts: List[str], semaphore: asyncio.Semaphore) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        async with semaphore:
            try:
                # Use asyncio.to_thread for non-async Together client
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=texts,
                    model=self.model
                )
                
                embeddings = []
                for data in response.data:
                    embeddings.append(data.embedding)
                
                return embeddings
                
            except Exception as e:
                print(f"Embedding generation failed for batch: {e}")
                # Return zero embeddings as fallback
                return [[0.0] * 1024] * len(texts)  # BGE-large has 1024 dimensions
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return [0.0] * 1024  # Return zero embedding
        
        embeddings = await self.generate_embeddings([text.strip()])
        return embeddings[0] if embeddings else [0.0] * 1024
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query - optimized for search"""
        if not query or not query.strip():
            return [0.0] * 1024
        
        # Clean and preprocess query
        cleaned_query = self._preprocess_query(query)
        
        # Generate embedding
        return await self.generate_single_embedding(cleaned_query)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better embedding quality"""
        # Remove excessive whitespace
        query = " ".join(query.split())
        
        # Ensure proper capitalization
        if query and not query[0].isupper():
            query = query[0].upper() + query[1:] if len(query) > 1 else query.upper()
        
        # Add question mark if it's clearly a question but missing punctuation
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'does', 'do', 'can', 'will']
        if any(query.lower().startswith(word) for word in question_words) and not query.endswith('?'):
            query += '?'
        
        return query
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text as cache key
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        try:
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def find_most_similar_chunks(self, query_embedding: List[float], chunk_embeddings: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar chunks to a query embedding"""
        if not query_embedding or not chunk_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(chunk_embeddings):
            embedding = chunk.get("embedding", [])
            if embedding:
                similarity = await self.calculate_similarity(query_embedding, embedding)
                similarities.append((similarity, i, chunk))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k chunks with similarity scores
        result = []
        for similarity, idx, chunk in similarities[:top_k]:
            enhanced_chunk = chunk.copy()
            enhanced_chunk["similarity_score"] = similarity
            result.append(enhanced_chunk)
        
        return result
    
    async def batch_similarity_search(self, query_embeddings: List[List[float]], chunk_embeddings: List[Dict[str, Any]], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform similarity search for multiple queries in parallel"""
        if not query_embeddings or not chunk_embeddings:
            return []
        
        # Process all queries in parallel
        tasks = [
            self.find_most_similar_chunks(query_embedding, chunk_embeddings, top_k)
            for query_embedding in query_embeddings
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Similarity search failed: {result}")
                final_results.append([])
            else:
                final_results.append(result)
        
        return final_results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check embedding service health"""
        try:
            # Test embedding generation
            start_time = time.time()
            test_embeddings = await self.generate_embeddings(["Health check test"])
            embedding_time = time.time() - start_time
            
            # Check embedding quality
            if test_embeddings and len(test_embeddings[0]) == 1024:
                embedding_status = "healthy"
            else:
                embedding_status = "degraded"
            
            return {
                "status": embedding_status,
                "model": self.model,
                "embedding_dimension": 1024,
                "test_time": embedding_time,
                "cache_enabled": self.embedding_cache is not None,
                "cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
                "batch_size": self.batch_size,
                "max_concurrent": self.max_concurrent
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            "model": self.model,
            "embedding_dimension": 1024,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "cache_enabled": self.embedding_cache is not None,
            "cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
            "cache_ttl": settings.cache_ttl if self.embedding_cache else None
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.embedding_cache:
            self.embedding_cache.clear()
            return True
        return False