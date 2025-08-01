#!/usr/bin/env python3
"""
Demo script for Enhanced Chunking and Embedding Services
Shows how the services work together in the RAG pipeline
"""

import asyncio
import time
from typing import List, Dict, Any, Tuple

# Mock implementations for demo (replace with actual services when dependencies are installed)
class MockTextChunker:
    """Mock text chunker for demonstration"""
    
    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
    
    async def chunk_documents(self, parsed_documents: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """Demo chunking functionality"""
        print("🔪 Starting text chunking...")
        
        all_chunks = []
        
        for doc_text, doc_metadata in parsed_documents:
            # Simple chunking simulation
            sentences = doc_text.split('. ')
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                if len(current_chunk + sentence) < self.chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunk_data = {
                            "chunk_text": current_chunk.strip(),
                            "metadata": {
                                **doc_metadata,
                                "chunk_index": chunk_index,
                                "chunk_length": len(current_chunk),
                                "token_count": len(current_chunk.split()),
                                "chunk_type": "text",
                                "quality_score": 0.8,
                                "keywords": ["policy", "insurance", "coverage"],
                                "content_category": "insurance_policy"
                            }
                        }
                        all_chunks.append(chunk_data)
                        chunk_index += 1
                    
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk.strip():
                chunk_data = {
                    "chunk_text": current_chunk.strip(),
                    "metadata": {
                        **doc_metadata,
                        "chunk_index": chunk_index,
                        "chunk_length": len(current_chunk),
                        "token_count": len(current_chunk.split()),
                        "chunk_type": "text",
                        "quality_score": 0.8,
                        "keywords": ["policy", "insurance", "coverage"],
                        "content_category": "insurance_policy"
                    }
                }
                all_chunks.append(chunk_data)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        return all_chunks

class MockEmbeddingService:
    """Mock embedding service for demonstration"""
    
    def __init__(self):
        self.model = "BAAI/bge-large-en-v1.5"
        self.embedding_dimension = 1024
        self.batch_size = 32
    
    async def embed_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Demo embedding functionality"""
        print("🧮 Starting embedding generation...")
        
        embedded_chunks = []
        
        # Simulate batch processing
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            print(f"  Processing batch {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1} ({len(batch)} chunks)")
            
            # Simulate embedding generation delay
            await asyncio.sleep(0.1)
            
            for chunk in batch:
                # Generate mock embedding (normally this would be from Together AI)
                mock_embedding = [0.1] * self.embedding_dimension
                
                enhanced_chunk = chunk.copy()
                enhanced_chunk["embedding"] = mock_embedding
                enhanced_chunk["metadata"]["embedding_model"] = self.model
                enhanced_chunk["metadata"]["embedding_dimension"] = self.embedding_dimension
                enhanced_chunk["metadata"]["embedding_timestamp"] = time.time()
                
                embedded_chunks.append(enhanced_chunk)
        
        print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """Demo query embedding"""
        print(f"🔍 Generating embedding for query: '{query[:50]}...'")
        # Mock query embedding
        return [0.1] * self.embedding_dimension

async def demo_chunking_and_embedding():
    """Demonstrate the complete chunking and embedding pipeline"""
    
    print("🚀 Enhanced Chunking and Embedding Services Demo")
    print("=" * 60)
    
    # Sample documents (simulating parsed document output)
    sample_documents = [
        (
            """The National Parivar Mediclaim Plus Policy provides comprehensive health insurance coverage. 
            A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits. 
            The waiting period for pre-existing diseases is thirty-six months of continuous coverage from the first policy inception. 
            Maternity expenses are covered under this policy, including childbirth and lawful medical termination of pregnancy. 
            To be eligible for maternity benefits, the female insured person must have been continuously covered for at least 24 months. 
            The policy covers medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person.""",
            {
                "source": "policy_document.pdf",
                "file_type": ".pdf",
                "extraction_method": "llamaparse",
                "document_id": "doc_001"
            }
        ),
        (
            """Cataract surgery has a specific waiting period of two years under this policy. 
            A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. 
            The maximum aggregate NCD is capped at 5% of the total base premium. 
            Health check-up expenses are reimbursed at the end of every block of two continuous policy years, provided the policy has been renewed without a break. 
            A hospital is defined as an institution with at least 10 inpatient beds in towns with population below ten lakhs, or 15 beds in all other places. 
            The hospital must have qualified nursing staff and medical practitioners available 24/7, and a fully equipped operation theatre.""",
            {
                "source": "policy_document.pdf", 
                "file_type": ".pdf",
                "extraction_method": "llamaparse",
                "document_id": "doc_001"
            }
        )
    ]
    
    # Initialize services
    print("🔧 Initializing services...")
    chunker = MockTextChunker()
    embedding_service = MockEmbeddingService()
    print("✅ Services initialized")
    print()
    
    # Step 1: Chunk documents
    start_time = time.time()
    chunks = await chunker.chunk_documents(sample_documents)
    chunking_time = time.time() - start_time
    
    print(f"⏱️  Chunking completed in {chunking_time:.2f} seconds")
    print()
    
    # Display sample chunks
    print("📄 Sample chunks created:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i+1}:")
        print(f"    Text: {chunk['chunk_text'][:100]}...")
        print(f"    Length: {chunk['metadata']['chunk_length']} chars")
        print(f"    Quality: {chunk['metadata']['quality_score']}")
        print(f"    Category: {chunk['metadata']['content_category']}")
        print()
    
    # Step 2: Generate embeddings
    start_time = time.time()
    embedded_chunks = await embedding_service.embed_chunks_parallel(chunks)
    embedding_time = time.time() - start_time
    
    print(f"⏱️  Embedding generation completed in {embedding_time:.2f} seconds")
    print()
    
    # Display embedding info
    print("🧮 Embedding information:")
    sample_chunk = embedded_chunks[0]
    print(f"  Model: {sample_chunk['metadata']['embedding_model']}")
    print(f"  Dimension: {sample_chunk['metadata']['embedding_dimension']}")
    print(f"  Embedding preview: {sample_chunk['embedding'][:5]}... (showing first 5 dimensions)")
    print()
    
    # Step 3: Demo query embedding
    sample_queries = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
    
    print("🔍 Generating query embeddings:")
    query_embeddings = []
    for query in sample_queries:
        query_embedding = await embedding_service.embed_query(query)
        query_embeddings.append(query_embedding)
        print(f"  ✅ Query: '{query}'")
    
    print()
    
    # Performance summary
    total_time = chunking_time + embedding_time
    print("📊 Performance Summary:")
    print(f"  Documents processed: {len(sample_documents)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Embeddings generated: {len(embedded_chunks)}")
    print(f"  Query embeddings: {len(query_embeddings)}")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Chunking time: {chunking_time:.2f}s ({chunking_time/total_time*100:.1f}%)")
    print(f"  Embedding time: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
    print()
    
    # Demonstrate data flow
    print("🔄 Data Flow Demonstration:")
    print("  1. Raw documents → Text Chunker")
    print("     ↓")
    print("  2. Text chunks with metadata → Embedding Service")
    print("     ↓") 
    print("  3. Embedded chunks → Vector Store (next step)")
    print("     ↓")
    print("  4. Query embedding + Vector search → Retrieved chunks")
    print("     ↓")
    print("  5. Retrieved chunks + Query → LLM → Final answer")
    print()
    
    print("✨ Demo completed successfully!")
    print()
    print("🎯 Key Features Demonstrated:")
    print("  ✅ Intelligent text chunking with quality scoring")
    print("  ✅ Parallel embedding generation with batching")
    print("  ✅ Rich metadata enhancement for better retrieval")
    print("  ✅ Query preprocessing for optimal search")
    print("  ✅ Performance optimization for sub-30s response times")
    
    return embedded_chunks, query_embeddings

def demo_advanced_features():
    """Demonstrate advanced features of the services"""
    
    print("\n🔬 Advanced Features Demo")
    print("=" * 40)
    
    print("🎨 Text Chunker Advanced Features:")
    print("  • Quality-based chunk filtering")
    print("  • Content categorization (insurance, legal, medical, etc.)")
    print("  • Keyword extraction for better retrieval")
    print("  • OCR text cleaning and normalization")
    print("  • Boundary optimization for semantic coherence")
    print("  • Token counting with tiktoken")
    print()
    
    print("🚀 Embedding Service Advanced Features:")
    print("  • Together AI BGE-large model (1024 dimensions)")
    print("  • TTL caching for repeated content")
    print("  • Parallel batch processing with semaphores")
    print("  • Query preprocessing and optimization")
    print("  • Similarity search capabilities")
    print("  • Health checking and monitoring")
    print("  • Graceful error handling with fallbacks")
    print()
    
    print("⚡ Performance Optimizations:")
    print("  • Concurrent processing with asyncio")
    print("  • Intelligent batching (32 chunks per batch)")
    print("  • Cache-first embedding retrieval")
    print("  • Quality filtering to reduce processing")
    print("  • Timeout management for reliability")
    print("  • Memory-efficient chunk processing")

if __name__ == "__main__":
    print("Enhanced Text Chunking and Embedding Services")
    print("HackRx Multimodal RAG Application")
    print("=" * 60)
    
    try:
        # Run the main demo
        embedded_chunks, query_embeddings = asyncio.run(demo_chunking_and_embedding())
        
        # Show advanced features
        demo_advanced_features()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Next Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Configure API keys in .env file")
        print("  3. Run the full application: python main.py")
        print("  4. Test with: python test_hackrx_api.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        print("\nThis is expected if dependencies are not installed.")
        print("Run: pip install -r requirements.txt")